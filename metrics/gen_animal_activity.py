import argparse
import os
import sys
from typing import List

import torch

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

from sampling.all import sample_all
from utils.setup import prepare_accelerator, prepare_pipeline
from metrics.gen_drawbench import json_to_configdict

animals = ['cat', 'dog', 'horse', 'monkey', 'rabbit', 'zebra', 'spider', 'bird', 'sheep', 'deer', 'cow', 'goat', 'lion',
           'tiger', 'bear', 'raccoon', 'fox', 'wolf', 'lizard', 'beetle', 'ant', 'butterfly', 'fish', 'shark', 'whale',
           'dolphin', 'squirrel', 'mouse', 'rat', 'snake', 'turtle', 'frog', 'chicken', 'duck', 'goose', 'bee', 'pig',
           'turkey', 'fly', 'llama', 'camel', 'bat', 'gorilla', 'hedgehog', 'kangaroo']
activities = ['riding a bike', 'playing chess', 'washing dishes']
print(animals[0].split())


def get_animal_activities_prompts() -> List[str]:
    prompts = []
    for animal in animals:
        for activity in activities:
            article = 'an' if animal[0] in 'aeuo' else 'a'
            prompts.append(f'{article} {animal} {activity}')
    return prompts


def get_animal_activities_items(prompts) -> List[List[int]]:
    item_idx = [[1, len(prompt.split()) - 1] for prompt in prompts]
    return item_idx


def generate_animal_activities_images(config_path):
    exp_dir = os.path.dirname(config_path)
    exp_name = os.path.basename(exp_dir)
    save_dir = os.path.join(exp_dir, 'animal_activities')
    print(f'Generate animal activities, experiment: {exp_name}')

    config = json_to_configdict(config_path)
    config.sample.batch_size = 1
    if 'base' not in exp_name:
        config.sample.finetuned_model = os.path.join(exp_dir, 'models_state_dict/model_50.pth')
    accelerator = prepare_accelerator(config, save_dir)
    pipeline = prepare_pipeline(config, accelerator, finetuning=False)

    config.prompt = get_animal_activities_prompts()
    config.item_idx = get_animal_activities_items(config.prompt)
    config.item_k = [[0.7] * len(items) for items in config.item_idx]

    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    sample_all(config, accelerator, pipeline, save_dir=None, img_save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate Animal Activities images for a saved config")
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    generate_animal_activities_images(args.config_path)


if __name__ == "__main__":
    main()
