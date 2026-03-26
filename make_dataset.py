from datasets import load_dataset


def count_words(prompt):
    if prompt is None:
        return 0
    return len(prompt.strip().split())


def make_dataset(data_path, name, save_path, to_filter=True):
    dataset = load_dataset(data_path, name, split='train')
    if ('prompt' not in dataset.column_names) and ('caption' in dataset.column_names):
        dataset= dataset.rename_column('caption', 'prompt')
    columns_to_remove = [col for col in dataset.column_names if col not in ['image', 'prompt']]
    dataset = dataset.remove_columns(columns_to_remove)
    if to_filter:
        dataset = dataset.filter(lambda row: 0 < count_words(row['prompt']) <= 15)
    dataset.save_to_disk(save_path)


make_dataset('poloclub/diffusiondb', '2m_first_10k', '/home/ergrishina_2/Diploma/diffusiondb')
make_dataset('mlx-community/dreambooth-dog6', None, '/home/ergrishina_2/Diploma/dog6')
make_dataset('Mercity/laion-subset', None, '/home/ergrishina_2/Diploma/laion', to_filter=False)
