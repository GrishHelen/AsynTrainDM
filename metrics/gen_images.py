import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from ml_collections import ConfigDict

script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
sys.path.append(project_root)

from sampling.all import sample_all
from utils.setup import prepare_accelerator, prepare_pipeline

PROMPT_CONFIG_DIR = os.path.join(project_root, 'config', 'prompt')


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


def generate_images(config_path: str, dataset_type: str):
    dataset_type = resolve_dataset_type(dataset_type)
    exp_dir = os.path.dirname(config_path)
    exp_name = os.path.basename(exp_dir)
    save_dir = os.path.join(exp_dir, dataset_type)
    print(f'Generate {dataset_type}, experiment: {exp_name}')

    config = json_to_configdict(config_path)
    config.sample.batch_size = 1
    if 'base' not in exp_name:
        config.sample.finetuned_model = os.path.join(exp_dir, 'models_state_dict/model_50.pth')
    accelerator = prepare_accelerator(config, save_dir)
    pipeline = prepare_pipeline(config, accelerator, finetuning=False)

    config.prompt = get_dataset_prompts(dataset_type)
    config.item_idx, config.item_k = get_dataset_items(dataset_type, config.prompt)

    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    sample_all(config, accelerator, pipeline, save_dir=None, img_save_dir=save_dir)


def main():
    available_datasets = ', '.join(sorted(get_available_datasets().values()))
    parser = argparse.ArgumentParser(description="Generate images for prompts from config/prompt")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_type", "--dataset", dest="dataset_type", type=str, required=True,
                        help=f"Dataset name from config/prompt. Available datasets: {available_datasets}",
    )
    args = parser.parse_args()
    generate_images(args.config_path, args.dataset_type)


if __name__ == "__main__":
    main()
