import copy
import gc
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler

from metrics.compute_metrics import evaluate_prompt_samples, load_images_from_path
from metrics.gen_images import (
    configure_prompt_set, generate_metric_images, get_dataset_items, get_dataset_prompts,
    get_samples_per_prompt, resolve_dataset_type,
)


def get_metrics_datasets(config):
    names = config.logging.get("metrics_datasets", [])
    if isinstance(names, str):
        names = names.replace(",", " ").split()
    datasets = list(dict.fromkeys(resolve_dataset_type(name) for name in names))
    if not datasets:
        raise ValueError("metrics_datasets must contain at least one validation prompt set")
    for name in datasets:
        prompts = get_dataset_prompts(name)
        if not prompts:
            raise ValueError(f"Validation prompt set {name!r} is empty")
        get_dataset_items(name, prompts)
    return datasets


def metrics_due(config, epoch):
    interval = int(config.logging.get("metrics_epoch", 0))
    if interval < 0:
        raise ValueError("metrics_epoch must be nonnegative")
    return interval > 0 and (epoch + 1) % interval == 0


@contextmanager
def metric_evaluation_state(pipeline, accelerator):
    old_scheduler = pipeline.scheduler
    old_unet = pipeline.unet
    was_training = old_unet.training
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cudnn_state = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        # Sampling resets global seeds; preserve the training RNG streams.
        with torch.random.fork_rng():
            pipeline.unet = accelerator.unwrap_model(old_unet)
            pipeline.unet.eval()
            pipeline.scheduler = DDIMScheduler.from_config(old_scheduler.config)
            yield
    finally:
        pipeline.scheduler = old_scheduler
        pipeline.unet = old_unet
        old_unet.train(was_training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cudnn_state


def release_metric_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def offload_for_metrics(pipeline, accelerator):
    models = []
    # DDP reducers own device-specific buffers, so do not move their parameters.
    if accelerator.num_processes == 1 and accelerator.device.type == "cuda":
        models = [(model, next(model.parameters()).device)
                  for model in (pipeline.unet, pipeline.vae, pipeline.text_encoder)]
    try:
        for model, _ in models:
            model.to("cpu")
        release_metric_memory()
        yield
    finally:
        release_metric_memory()
        for model, device in models:
            model.to(device)


def make_metric_evaluator(metric, logging_config, device):
    if metric == "clip":
        from metrics.clip_score import CLIPScoreEvaluator

        return CLIPScoreEvaluator(
            device=device, batch_size=32, model_id=logging_config.get("metrics_clip_model", None)
        )
    from metrics.qwen_score import QwenScoreEvaluator

    return QwenScoreEvaluator(device=device, model_id=logging_config.get("metrics_qwen_model", None))


def save_metric_record(path, record):
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    key = (record["epoch"], record["dataset"], record["sampler"])
    records = [row for row in records if (row["epoch"], row["dataset"], row["sampler"]) != key]
    records.append(record)
    staging = path.with_suffix(".json.tmp")
    staging.write_text(json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    staging.replace(path)


def evaluate_epoch_metrics(config, accelerator, pipeline, save_dir, epoch):
    if not metrics_due(config, epoch):
        return []
    accelerator.wait_for_everyone()
    try:
        if not accelerator.is_main_process:
            return []
        datasets = get_metrics_datasets(config)
        samples_per_prompt = get_samples_per_prompt(config)

        epoch_number = epoch + 1
        save_dir = Path(save_dir)
        metrics_path = save_dir / "metrics.json"
        eval_config = copy.deepcopy(config)
        samplers = ["AsynDM"]
        if eval_config.generate_dm:
            samplers.insert(0, "DM")
        if eval_config.generate_dm_concave:
            samplers.append("dm_concave")
        device = config.logging.get("metrics_device", "auto")
        if device == "auto":
            device = str(accelerator.device)
        jobs = []
        records = {}

        with metric_evaluation_state(pipeline, accelerator):
            with accelerator.autocast(), torch.no_grad():
                for dataset in datasets:
                    configure_prompt_set(eval_config, dataset)
                    image_dir = save_dir / "eval_images" / f"epoch_{epoch_number}" / dataset
                    image_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[Metrics] Epoch {epoch_number}: generating {dataset}", flush=True)
                    generate_metric_images(eval_config, accelerator, pipeline, img_save_dir=str(image_dir),
                                           samples_per_prompt=samples_per_prompt)
                    jobs.append((dataset, image_dir, list(eval_config.prompt)))

            with offload_for_metrics(pipeline, accelerator):
                # Load each evaluator once per epoch, and keep only one on the device.
                for metric in ("clip", "qwen"):
                    evaluator = None
                    try:
                        evaluator = make_metric_evaluator(metric, config.logging, device)
                        for dataset, image_dir, prompts in jobs:
                            expected_count = len(prompts) * samples_per_prompt
                            images_by_sampler = load_images_from_path(str(image_dir), expected_count=expected_count)
                            try:
                                for sampler in samplers:
                                    images = images_by_sampler[sampler]
                                    if len(images) != expected_count:
                                        raise ValueError(f"Missing {sampler} images for {dataset} in {image_dir}")
                                    score, per_prompt = evaluate_prompt_samples(evaluator, images, prompts, samples_per_prompt)
                                    record = records.setdefault((dataset, sampler), {
                                        "epoch": epoch_number,
                                        "dataset": dataset,
                                        "sampler": sampler,
                                        "num_images": len(images),
                                        "num_prompts": len(prompts),
                                        "samples_per_prompt": samples_per_prompt,
                                        "images_dir": image_dir.relative_to(save_dir).as_posix(),
                                    })
                                    record[metric] = score
                                    save_metric_record(metrics_path, record)
                                    if metric == "qwen":
                                        print(
                                            f"[Metrics] Epoch {epoch_number} | {dataset} | {sampler} | "
                                            f"CLIP={record['clip']:.6f} | Qwen={record['qwen']:.6f}",
                                            flush=True,
                                        )
                                    else:
                                        print(
                                            f"[Metrics] Epoch {epoch_number} | {dataset} | {sampler} | "
                                            f"CLIP={score:.6f}", flush=True,
                                        )
                            finally:
                                for images in images_by_sampler.values():
                                    for image in images:
                                        image.close()
                    finally:
                        del evaluator
                        release_metric_memory()
        print(f"[Metrics] Saved to {metrics_path}", flush=True)
        return list(records.values())
    finally:
        accelerator.wait_for_everyone()
