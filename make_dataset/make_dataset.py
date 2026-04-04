import argparse

import datasets
from datasets import load_dataset

from segmentation import config, masks_for_dataset


def count_words(prompt):
    if prompt is None:
        return 0
    return len(prompt.strip().split())


def make_dataset(data_path, name, save_path, to_filter=True, max_samples=-1, make_masks=False):
    if data_path is None:
        return
    dataset = load_dataset(data_path, name, split='train')
    if ('prompt' not in dataset.column_names) and ('caption' in dataset.column_names):
        dataset = dataset.rename_column('caption', 'prompt')
    columns_to_remove = [col for col in dataset.column_names if col not in ['image', 'prompt']]
    dataset = dataset.remove_columns(columns_to_remove)
    if to_filter:
        dataset = dataset.filter(lambda row: 0 < count_words(row['prompt']) <= 15)
    if 0 < max_samples < len(dataset):
        dataset = dataset.train_test_split(train_size=max_samples, shuffle=False, seed=1234)['train']

    if make_masks:
        masks = masks_for_dataset(dataset, config)
        dataset_masks = datasets.Dataset.from_dict({"mask": masks})
        dataset = datasets.concatenate_datasets([dataset, dataset_masks], axis=1)

    dataset.save_to_disk(save_path)


# make_dataset('poloclub/diffusiondb', '2m_first_10k', '/home/ergrishina_2/Diploma/diffusiondb')
# make_dataset('mlx-community/dreambooth-dog6', None, '/home/ergrishina_2/Diploma/dog6')
# make_dataset('Mercity/laion-subset', None, '/home/ergrishina_2/Diploma/laion', to_filter=False)
# make_dataset('Mercity/laion-subset', None, '/home/ergrishina_2/Diploma/laion_3k',
#              to_filter=False, max_samples=3000)
# make_dataset('Mercity/laion-subset', None, '/home/ergrishina_2/Diploma/laion_3k_masks',
#              to_filter=False, max_samples=3000, make_masks=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing arguments from console")

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--to_filter", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--make_masks", type=int, default=0)

    args = parser.parse_args()
    make_dataset(args.data_path, args.name, args.save_path, args.to_filter, args.max_samples, args.make_masks)
