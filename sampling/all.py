import json
import os
import sys

from tqdm import tqdm

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

from sampling.base import generate_dm, generate_dm_concave
from sampling.asyn import generate_asyn
from utils.utils import seed_everything
from utils.sampling import prepare_encoded_prompts, encode_prompts_list


def sample_all(config, accelerator, pipeline, save_dir=None, img_save_dir=None):
    # generate negative prompt embeddings
    neg_prompt_embed = encode_prompts_list(pipeline, accelerator.device, [""])
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)

    prompt_list = [config.prompt] if isinstance(config.prompt, str) else config.prompt
    if len(config.prompt_file) != 0:
        with open(config.prompt_file, 'r') as f:
            prompt_list = json.load(f)
    # print('prompt list:', prompt_list)
    prompt_cnt = len(prompt_list)
    total_num_batches_per_epoch = config.sample.num_batches_per_epoch * prompt_cnt

    pipeline.unet.eval()
    total_prompts1 = []
    global_idx = config.begin_index * config.sample.batch_size
    if global_idx and save_dir is not None:
        with open(os.path.join(save_dir, f'prompt.json'), 'r') as f:
            total_prompts1 = json.load(f)[:global_idx]
    if img_save_dir is None:
        img_save_dir = os.path.join(save_dir, "images/")
    for idx in tqdm(
            range(config.begin_index, total_num_batches_per_epoch),
            disable=not accelerator.is_local_main_process,
            position=0,
    ):
        seed_everything(config.seed)
        # generate prompts
        prompt_idx = idx // config.sample.num_batches_per_epoch
        total_prompts1.extend([[prompt_list[prompt_idx] for _ in range(config.sample.batch_size)]])
        prompt_embeds1_combine = prepare_encoded_prompts(config, accelerator, pipeline, prompt_list[prompt_idx],
                                                         sample_neg_prompt_embeds)
        cross_mask = None

        # ================================================================= #
        # base (DM)
        if config.generate_dm or config.static_mask:
            cross_mask = generate_dm(config, accelerator, pipeline, idx, prompt_embeds1_combine, img_save_dir)

        # ================================================================= #
        # base2 (DM concave)
        if config.generate_dm_concave:
            generate_dm_concave(config, accelerator, pipeline, idx, prompt_embeds1_combine, img_save_dir)

        # ================================================================= #
        # asyndm
        generate_asyn(config, accelerator, pipeline, idx, prompt_embeds1_combine, cross_mask, img_save_dir)

        global_idx += config.sample.batch_size
