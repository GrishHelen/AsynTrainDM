from datasets import load_dataset


def count_words(prompt):
    if prompt is None:
        return 0
    return len(prompt.strip().split())


dataset = load_dataset('poloclub/diffusiondb', '2m_first_10k', split='train')

columns_to_remove = [col for col in dataset.column_names if col not in ['image', 'prompt']]
dataset = dataset.remove_columns(columns_to_remove)

filtered_dataset = dataset.filter(lambda row: 0 < count_words(row['prompt']) <= 15)
filtered_dataset.save_to_disk('/home/ergrishina_2/Diploma/diffusiondb')
