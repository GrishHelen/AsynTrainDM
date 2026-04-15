import argparse
import csv
import json
import os
import sys
from typing import List

import spacy
import torch
from datasets import load_dataset
from ml_collections import ConfigDict
from tqdm import tqdm

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

from sampling.all import sample_all
from utils.setup import prepare_accelerator, prepare_pipeline

DRAWBENCH_ITEMS_PATH = "/home/ergrishina_2/Diploma/AsynDM/metrics/drawbench_item_idx.csv"


def json_to_configdict(json_path: str) -> ConfigDict:
    def _convert_to_configdict(obj):
        if isinstance(obj, dict):
            if "_value_" in obj and "_name_" in obj and "__objclass__" in obj:
                return obj["_value_"]
            else:
                return ConfigDict({k: _convert_to_configdict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [_convert_to_configdict(item) for item in obj]
        else:
            return obj

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _convert_to_configdict(data)


def get_drawbench_prompts() -> List[str]:
    prompts = [item['prompts'] for item in load_dataset("shunk031/DrawBench")['test']]
    return prompts


def get_drawbench_items() -> List[List[int]]:
    item_idx = []
    with open(DRAWBENCH_ITEMS_PATH, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            item_idx.append(list(map(int, row)))
    return item_idx


def extract_objects_with_spacy(prompts: List[str]):
    """
    Extracts indices of words that correspond to object nouns.
    """
    nlp = spacy.load("en_core_web_sm")
    item_idx = []
    for prompt in tqdm(prompts):
        doc = nlp(prompt)
        words = prompt.split()
        word_boundaries = []  # list of (start_char, end_char, word_index) for each word
        pos = 0
        for idx, word in enumerate(words):
            start = prompt.find(word, pos)
            end = start + len(word)
            word_boundaries.append((start, end, idx))
            pos = end
        # Map each token to a word index (if token's start falls inside a word)
        token_to_word_idx = {}
        for token in doc:
            token_start = token.idx
            for start, end, w_idx in word_boundaries:
                if start <= token_start < end:
                    token_to_word_idx[token.i] = w_idx
                    break
        # Collect unique word indices for nouns/proper nouns (excluding stopwords)
        object_word_indices = set()
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and token.dep_ in ["ROOT", "nsubj", "dobj", "conj"]:
                if token.is_stop:
                    continue
                if token.i in token_to_word_idx:
                    object_word_indices.add(token_to_word_idx[token.i])
        item_idx.append(sorted(list(object_word_indices)))
    return item_idx


def generate_drawbench_images(config_path):
    exp_dir = os.path.dirname(config_path)
    exp_name = os.path.basename(exp_dir)
    save_dir = os.path.join(exp_dir, 'drawbench')
    print(f'Generate drawbench, experiment: {exp_name}')

    config = json_to_configdict(config_path)
    config.sample.batch_size = 1
    if 'base' not in exp_name:
        config.sample.finetuned_model = os.path.join(exp_dir, 'models_state_dict/model_50.pth')
    accelerator = prepare_accelerator(config, save_dir)
    pipeline = prepare_pipeline(config, accelerator, finetuning=False)

    config.prompt = get_drawbench_prompts()
    config.item_idx = get_drawbench_items()
    config.item_k = [[0.7] * len(items) for items in config.item_idx]

    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    sample_all(config, accelerator, pipeline, save_dir=None, img_save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate DrawBench images for a saved config")
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    generate_drawbench_images(args.config_path)


if __name__ == "__main__":
    main()
