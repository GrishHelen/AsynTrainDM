import argparse
import json
import os
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

from metrics.gen_images import resolve_dataset_type, get_dataset_prompts, get_samples_per_prompt


class MetricType(Enum):
    QWEN = 'qwen'
    CLIP = 'clip'


def load_generation_info(img_folder):
    path = Path(img_folder) / "generation.json"
    if not path.exists():
        return None
    info = json.loads(path.read_text(encoding="utf-8"))
    if not info.get("complete"):
        raise ValueError(f"Metric image generation is incomplete in {img_folder}")
    if info.get("order") != "round_major":
        raise ValueError(f"Unknown metric image order in {path}")
    get_samples_per_prompt({}, info["samples_per_prompt"])
    if not isinstance(info["prompts"], list) or not info["prompts"]:
        raise ValueError(f"Expected a nonempty prompt list in {path}")
    return info


def evaluate_prompt_samples(evaluator, images, prompts, samples_per_prompt=1):
    count = get_samples_per_prompt({}, samples_per_prompt)
    if not prompts or len(images) != len(prompts) * count:
        raise ValueError("Expected exactly samples_per_prompt images for every prompt")
    # Image indices are round-major: the entire prompt list repeats for each round.
    scores = np.asarray(evaluator.evaluate_scores(images, prompts * count), dtype=np.float64)
    if scores.shape != (len(images),) or not np.isfinite(scores).all():
        raise ValueError("The evaluator must return one finite score per image")
    per_prompt = scores.reshape(count, len(prompts)).mean(axis=0)
    return float(per_prompt.mean()), per_prompt.tolist()


def load_images_from_path(img_folder: str, expected_count=None) -> Dict[str, List[Image.Image]]:
    info = load_generation_info(img_folder)
    if info is not None:
        total = len(info["prompts"]) * info["samples_per_prompt"]
        if expected_count is not None and expected_count != total:
            raise ValueError(f"Expected {expected_count} images per sampler, but {img_folder} declares {total}")
        expected_count = total
    images_by_method: Dict[str, List[Tuple[int, Image.Image]]] = {method: [] for method in
                                                                  ("DM", "dm_concave", "AsynDM")}
    for filename in os.listdir(img_folder):
        match = re.compile(r"^(\d{5,})_(DM|dm_concave|AsynDM)\.png$").match(filename)
        if not match:
            continue
        index_str, method = match.groups()
        # Ignore leftovers from an older, longer run or a now-disabled sampler.
        if info is not None and (method not in info["samplers"] or int(index_str) >= expected_count):
            continue
        full_path = os.path.join(img_folder, filename)
        with Image.open(full_path) as img:
            img.load()
            loaded_img = img.convert("RGB") if img.mode != "RGB" else img.copy()
            images_by_method[method].append((int(index_str), loaded_img))
    for method in images_by_method:
        images_by_method[method] = sorted(images_by_method[method], key=lambda x: x[0])
        required = info is not None and method in info["samplers"]
        if expected_count is not None and (images_by_method[method] or required):
            indices = [index for index, _ in images_by_method[method]]
            if indices != list(range(expected_count)):
                raise ValueError(
                    f"Expected image indices 0..{expected_count - 1} for {method} in {img_folder}; "
                    f"found {indices}"
                )
        images_by_method[method] = list(map(lambda x: x[1], images_by_method[method]))
    return images_by_method


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for generated images")
    parser.add_argument("--metric", type=str, default='qwen')
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--gen_method", type=str, default=None)
    parser.add_argument("--metrics_samples_per_prompt", "--samples_per_prompt", dest="samples_per_prompt",
                        type=int, default=None, help="Normally read from generation.json; defaults to 1 for old folders")
    args = parser.parse_args()

    dataset = os.path.basename(args.img_folder)
    if dataset == '':
        dataset = os.path.basename(os.path.dirname(args.img_folder))

    print(f"Metric to compute: {args.metric}. Dataset: {dataset}. img_folder: {args.img_folder}")

    info = load_generation_info(args.img_folder)
    if info is None:
        prompts = get_dataset_prompts(resolve_dataset_type(dataset))
        count = get_samples_per_prompt({}, args.samples_per_prompt)
    else:
        prompts = info["prompts"]
        count = info["samples_per_prompt"]
        if args.samples_per_prompt is not None and args.samples_per_prompt != count:
            raise ValueError("samples_per_prompt disagrees with generation.json")
    images_by_method = load_images_from_path(args.img_folder, expected_count=len(prompts) * count)
    scores_by_method = {}
    try:
        methods = [args.gen_method] if args.gen_method is not None else [
            method for method, images in images_by_method.items() if images
        ]
        if not methods or any(method not in images_by_method or not images_by_method[method] for method in methods):
            raise ValueError("No images found for the requested generation method(s)")
        if args.metric == MetricType.QWEN.value:
            from metrics.qwen_score import QwenScoreEvaluator

            evaluator = QwenScoreEvaluator(device=args.device, max_new_tokens=args.max_new_tokens, model_id=args.model_id)
        elif args.metric == MetricType.CLIP.value:
            from metrics.clip_score import CLIPScoreEvaluator

            evaluator = CLIPScoreEvaluator(device=args.device, batch_size=32, model_id=args.model_id)
        else:
            raise ValueError(f"Unknown metric '{args.metric}' to compute")

        for method in methods:
            score, _ = evaluate_prompt_samples(evaluator, images_by_method[method], prompts, count)
            scores_by_method[method] = score
            print(f'Method {method}. Score: {score}. Samples per prompt: {count}')
        print(f"img_folder: {args.img_folder}")
        print(json.dumps(scores_by_method, indent=2))
        return scores_by_method
    finally:
        for images in images_by_method.values():
            for image in images:
                image.close()


if __name__ == "__main__":
    main()
