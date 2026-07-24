import argparse
import json
import os
import re
import sys
from enum import Enum
from typing import Dict, List, Tuple

from PIL import Image

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

from metrics.clip_score import compute_clip_score
from metrics.qwen_score import compute_qwen_score
from metrics.gen_images import resolve_dataset_type, get_dataset_prompts


class MetricType(Enum):
    QWEN = 'qwen'
    CLIP = 'clip'


def load_images_from_path(img_folder: str) -> Dict[str, List[Image.Image]]:
    images_by_method: Dict[str, List[Tuple[int, Image.Image]]] = {method: [] for method in
                                                                  ("DM", "dm_concave", "AsynDM")}
    for filename in os.listdir(img_folder):
        match = re.compile(r"^(\d{5})_(DM|dm_concave|AsynDM)\.png$").match(filename)
        if not match:
            continue
        index_str, method = match.groups()
        full_path = os.path.join(img_folder, filename)
        with Image.open(full_path) as img:
            img.load()
            loaded_img = img.convert("RGB") if img.mode != "RGB" else img.copy()
            images_by_method[method].append((int(index_str), loaded_img))
    for method in images_by_method:
        images_by_method[method] = sorted(images_by_method[method], key=lambda x: x[0])
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
    args = parser.parse_args()

    dataset = os.path.basename(args.img_folder)
    if dataset == '':
        dataset = os.path.basename(os.path.dirname(args.img_folder))

    print(f"Metric to compute: {args.metric}. Dataset: {dataset}. img_folder: {args.img_folder}")

    dataset = resolve_dataset_type(dataset)
    prompts = get_dataset_prompts(dataset)
    images_by_method = load_images_from_path(args.img_folder)
    scores_by_method = {}

    if args.metric == MetricType.QWEN.value:
        print(f'Metric: {MetricType.QWEN.value}')

        if args.gen_method is not None:
            if args.gen_method not in images_by_method.keys():
                return
            score = compute_qwen_score(
                images_by_method[args.gen_method],
                prompts,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                model_id=args.model_id,
            )
            scores_by_method[args.gen_method] = score
            print(f'Method {args.gen_method}. Score: {score}')
            return

        for method in images_by_method.keys():
            score = compute_qwen_score(
                images_by_method[method],
                prompts,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                model_id=args.model_id,
            )
            scores_by_method[method] = score
            print(f'Method {method}. Score: {score}')

        print(f"img_folder: {args.img_folder}")
        print(json.dumps(scores_by_method, indent=2))
        return scores_by_method

    if args.metric == MetricType.CLIP.value:
        print(f'Metric: {MetricType.CLIP.value}')

        if args.gen_method is not None:
            if args.gen_method not in images_by_method.keys():
                return
            score = compute_clip_score(
                images_by_method[args.gen_method],
                prompts,
                device=args.device,
                model_id=args.model_id,
            )
            scores_by_method[args.gen_method] = score
            print(f'Method {args.gen_method}. Score: {score}')
            return

        for method in images_by_method.keys():
            if len(images_by_method[method]) == 0:
                continue
            score = compute_clip_score(
                images_by_method[method],
                prompts,
                device=args.device,
                model_id=args.model_id,
            )
            scores_by_method[method] = score
            print(f'Method {method}. Score: {score}')

        print(f"img_folder: {args.img_folder}")
        print(json.dumps(scores_by_method, indent=2))
        return scores_by_method

    raise ValueError(f"Unknown metric '{args.metric}' to compute")


if __name__ == "__main__":
    main()
