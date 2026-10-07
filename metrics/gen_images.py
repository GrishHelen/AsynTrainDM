import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from ml_collections import ConfigDict
from tqdm import tqdm

script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
sys.path.append(project_root)

PROMPT_CONFIG_DIR = os.path.join(project_root, 'config', 'prompt')


def resolve_config_path(config_path: str) -> str:
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return config_path


def json_to_configdict(json_path: str) -> ConfigDict:
    def _convert_to_configdict(obj):
        if isinstance(obj, dict):
            if "_value_" in obj and "_name_" in obj and "__objclass__" in obj:
                return obj["_value_"]
            return ConfigDict({k: _convert_to_configdict(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_convert_to_configdict(item) for item in obj]
        return obj

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _convert_to_configdict(data)


def get_available_datasets() -> Dict[str, str]:
    dataset_map = {}
    for filename in os.listdir(PROMPT_CONFIG_DIR):
        if filename.endswith('_item.json') or not filename.endswith('.json'):
            continue
        dataset_name = os.path.splitext(filename)[0]
        item_config_path = os.path.join(PROMPT_CONFIG_DIR, f'{dataset_name}_item.json')
        if os.path.exists(item_config_path):
            dataset_map[dataset_name.lower()] = dataset_name
    return dataset_map


def resolve_dataset_type(dataset_type: str) -> str:
    dataset_type = dataset_type.split(",", maxsplit=1)[0].strip()
    dataset_type = re.sub(r"_robust$", "", dataset_type, flags=re.IGNORECASE)
    dataset_map = get_available_datasets()
    resolved_dataset_type = dataset_map.get(dataset_type.lower())
    if resolved_dataset_type is None:
        available_datasets = ', '.join(sorted(dataset_map.values()))
        raise ValueError(
            f"Unknown dataset '{dataset_type}'. Available datasets: {available_datasets}"
        )
    return resolved_dataset_type


def get_dataset_prompts(dataset_type: str) -> List[str]:
    prompt_path = os.path.join(PROMPT_CONFIG_DIR, f'{dataset_type}.json')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    if not isinstance(prompts, list):
        raise ValueError(f"Prompt config '{prompt_path}' must contain a list of prompts")
    return prompts


def get_dataset_items(dataset_type: str, prompts: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
    item_path = os.path.join(PROMPT_CONFIG_DIR, f'{dataset_type}_item.json')
    with open(item_path, 'r', encoding='utf-8') as f:
        item_data = json.load(f)

    item_idx_by_prompt = item_data.get('item_idx')
    item_k_by_prompt = item_data.get('item_k', {})
    if not isinstance(item_idx_by_prompt, dict):
        raise ValueError(f"Item config '{item_path}' must contain an 'item_idx' mapping")
    if not isinstance(item_k_by_prompt, dict):
        raise ValueError(f"Item config '{item_path}' must contain an 'item_k' mapping")

    item_idx = []
    item_k = []

    for prompt in prompts:
        prompt_item_idx = item_idx_by_prompt.get(prompt)
        if prompt_item_idx is None:
            raise ValueError(f"Missing item_idx for prompt '{prompt}'")

        prompt_item_k = item_k_by_prompt.get(prompt)
        if prompt_item_k is None:
            raise ValueError(f"Missing item_k for prompt '{prompt}'")

        item_idx.append(prompt_item_idx)
        item_k.append(prompt_item_k)

    return item_idx, item_k


def resolve_finetuned_model_path(config: ConfigDict, exp_dir: str, exp_name: str, finetuned_model: Optional[str] = None):
    if finetuned_model is not None:
        return finetuned_model
    if 'base' in exp_name.lower():
        return None

    models_dir = os.path.join(exp_dir, 'models_state_dict')
    n_epochs = int(config.finetune.n_epochs) if 'finetune' in config and 'n_epochs' in config.finetune else None
    if n_epochs is not None:
        epoch_model = os.path.join(models_dir, f'model_{n_epochs}.pth')
        if os.path.exists(epoch_model):
            return epoch_model

    if os.path.isdir(models_dir):
        model_candidates = []
        for filename in os.listdir(models_dir):
            match = re.fullmatch(r'model_(\d+)\.pth', filename)
            if match:
                model_candidates.append((int(match.group(1)), os.path.join(models_dir, filename)))
        if model_candidates:
            return max(model_candidates, key=lambda item: item[0])[1]

    fallback = os.path.join(models_dir, 'model_50.pth')
    raise FileNotFoundError(
        f"Could not find finetuned checkpoint. Expected '{fallback}' or another 'model_*.pth' in '{models_dir}'. "
        "Pass --finetuned_model explicitly if the checkpoint is stored elsewhere."
    )


def apply_attention_grid_options(
        config: ConfigDict,
        save_attn_grids: bool,
        attn_grid_every: Optional[int],
        attn_grid_cell_size: Optional[int],
        mask_thr: Optional[float],
        attn_mask_threshold_type: Optional[str],
):
    config.sample.save_attn_grids = bool(save_attn_grids)
    if attn_grid_every is not None:
        config.sample.attn_grid_every = attn_grid_every
    elif 'attn_grid_every' not in config.sample:
        config.sample.attn_grid_every = 5

    if attn_grid_cell_size is not None:
        config.sample.attn_grid_cell_size = attn_grid_cell_size
    elif 'attn_grid_cell_size' not in config.sample:
        config.sample.attn_grid_cell_size = 192

    if mask_thr is not None:
        config.mask_thr = mask_thr

    if attn_mask_threshold_type is not None:
        attn_mask_threshold_type = attn_mask_threshold_type.lower()
        if attn_mask_threshold_type not in {"mean", "robust"}:
            raise ValueError(
                f"Unknown attention mask threshold type: {attn_mask_threshold_type!r}. "
                "Expected 'mean' or 'robust'."
            )
        config.sample.attn_mask_threshold_type = attn_mask_threshold_type
    elif 'attn_mask_threshold_type' not in config.sample:
        config.sample.attn_mask_threshold_type = "mean"


def configure_prompt_set(config: ConfigDict, dataset_type: str):
    config.prompt = get_dataset_prompts(dataset_type)
    config.item_idx, config.item_k = get_dataset_items(dataset_type, config.prompt)
    config.prompt_file = ""
    config.item_idx_file = ""


def get_samples_per_prompt(config, override=None):
    count = config.get("logging", {}).get("metrics_samples_per_prompt", 1) if override is None else override
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("metrics_samples_per_prompt must be a positive integer")
    return count


@torch.no_grad()
def generate_metric_images(config, accelerator, pipeline, img_save_dir, samples_per_prompt=None):
    from sampling.base import generate_dm, generate_dm_concave
    from sampling.asyn import generate_asyn
    from utils.sampling import encode_prompts_list, prepare_encoded_prompts
    from utils.utils import seed_everything

    count = get_samples_per_prompt(config, samples_per_prompt)
    prompts = list(config.prompt)
    if not prompts:
        raise ValueError("Expected at least one prompt for metric generation")
    eval_config = copy.deepcopy(config)
    # This private config does not change the batch size used for visual validation.
    eval_config.sample.batch_size = 1
    eval_config.sample.num_batches_per_epoch = 1
    eval_config.begin_index = 0
    eval_config.prompt_file = ""
    eval_config.item_idx_file = ""
    # Flat indices follow round * num_prompts + prompt_index, not prompt-major order.
    eval_config.prompt = prompts * count
    eval_config.item_idx = list(config.item_idx) * count
    eval_config.item_k = list(config.item_k) * count

    samplers = ["AsynDM"]
    if config.generate_dm:
        samplers.insert(0, "DM")
    if config.generate_dm_concave:
        samplers.append("dm_concave")
    output_dir = Path(img_save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "prompts": prompts, "samples_per_prompt": count, "seed": config.seed,
        "order": "round_major", "samplers": samplers, "complete": False,
    }
    manifest_path = output_dir / "generation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    pipeline.unet.eval()
    seed_everything(config.seed)
    # Separate streams give samplers the same starting noise without reseeding per image.
    generators = {method: [torch.Generator(device=accelerator.device).manual_seed(config.seed)]
                  for method in ("DM", "dm_concave", "AsynDM")}
    negative_embeds = encode_prompts_list(pipeline, accelerator.device, [""])
    for idx, prompt in enumerate(tqdm(
            eval_config.prompt, desc="Metric images", disable=not accelerator.is_local_main_process,
    )):
        prompt_embeds = prepare_encoded_prompts(eval_config, accelerator, pipeline, prompt, negative_embeds)
        cross_mask = None
        if eval_config.generate_dm or eval_config.static_mask:
            cross_mask = generate_dm(eval_config, accelerator, pipeline, idx, prompt_embeds,
                                     str(output_dir), generators=generators["DM"])
        if eval_config.generate_dm_concave:
            generate_dm_concave(eval_config, accelerator, pipeline, idx, prompt_embeds,
                                str(output_dir), generators=generators["dm_concave"])
        generate_asyn(eval_config, accelerator, pipeline, idx, prompt_embeds, cross_mask,
                      str(output_dir), prompt=prompt, generators=generators["AsynDM"])

    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_images(
        config_path: str,
        dataset_type: str,
        finetuned_model=None,
        save_attn_grids=False,
        attn_grid_every=None,
        attn_grid_cell_size=None,
        mask_thr=None,
        attn_mask_threshold_type=None,
        asyn_k=None,
        samples_per_prompt=None,
):
    config_path = resolve_config_path(config_path)
    dataset_type = resolve_dataset_type(dataset_type)
    exp_dir = os.path.dirname(config_path)
    exp_name = os.path.basename(exp_dir)

    config = json_to_configdict(config_path)
    samples_per_prompt = get_samples_per_prompt(config, samples_per_prompt)
    config.sample.finetuned_model = resolve_finetuned_model_path(config, exp_dir, exp_name, finetuned_model)
    apply_attention_grid_options(
        config,
        save_attn_grids,
        attn_grid_every,
        attn_grid_cell_size,
        mask_thr,
        attn_mask_threshold_type,
    )
    if asyn_k is not None:
        asyn_k = float(asyn_k)
        if not 0.0 <= asyn_k <= 1.0:
            raise ValueError(f"asyn_k must be in [0, 1], got {asyn_k}")
        config.sample.item_k = asyn_k

    effective_threshold_type = config.sample.get("attn_mask_threshold_type", "mean").lower()
    output_dataset_name = dataset_type
    if effective_threshold_type == "robust":
        output_dataset_name = f"{output_dataset_name}_robust"

    effective_item_k = config.sample.get("item_k", None)
    if effective_item_k is None:
        output_dir_name = output_dataset_name
    else:
        output_dir_name = f"{output_dataset_name}, k={float(effective_item_k):g}"
    save_dir = os.path.join(exp_dir, output_dir_name)
    print(f'Generate {output_dir_name}, experiment: {exp_name}')

    from utils.setup import prepare_accelerator, prepare_pipeline

    accelerator = prepare_accelerator(config, save_dir)
    pipeline = prepare_pipeline(config, accelerator, finetuning=False)

    configure_prompt_set(config, dataset_type)

    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    generate_metric_images(config, accelerator, pipeline, save_dir, samples_per_prompt=samples_per_prompt)


def main():
    available_datasets = ', '.join(sorted(get_available_datasets().values()))
    parser = argparse.ArgumentParser(description="Generate images for prompts from config/prompt")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to experiment config.json or to an experiment directory containing config.json")
    parser.add_argument("--dataset_type", "--dataset", "--prompt_set", dest="dataset_type", type=str, required=True,
                        help=f"Dataset name from config/prompt. Available datasets: {available_datasets}",
                        )
    parser.add_argument("--finetuned_model", "--finetuned", dest="finetuned_model", type=str, default=None,
                        help=f"Path to the finetuned model checkpoint",
                        )
    parser.add_argument("--save_attn_grids", "--save_cross_attention_grids", type=int, default=0)
    parser.add_argument("--attn_grid_every", "--cross_attention_grid_every", type=int, default=5)
    parser.add_argument("--attn_grid_cell_size", type=int, default=192)
    parser.add_argument("--mask_thr", type=float, default=1.0)
    parser.add_argument("--attn_mask_threshold_type", choices=["mean", "robust"], default="mean",
                        help="Cross-attention mask construction used by AsynDM inference")
    parser.add_argument("--asyn_k", type=float, default=None,
                        help="Override item_k for all objects during AsynDM inference (0=linear, 1=quadratic)")
    parser.add_argument("--metrics_samples_per_prompt", "--samples_per_prompt", dest="samples_per_prompt",
                        type=int, default=None, help="Images per prompt for metrics (config value or 1)")
    args = parser.parse_args()
    generate_images(
        args.config_path,
        args.dataset_type,
        finetuned_model=args.finetuned_model,
        save_attn_grids=bool(args.save_attn_grids),
        attn_grid_every=args.attn_grid_every,
        attn_grid_cell_size=args.attn_grid_cell_size,
        mask_thr=args.mask_thr,
        attn_mask_threshold_type=args.attn_mask_threshold_type,
        asyn_k=args.asyn_k,
        samples_per_prompt=args.samples_per_prompt,
    )


if __name__ == "__main__":
    main()
