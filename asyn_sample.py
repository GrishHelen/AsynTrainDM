import datetime
import json
import os
import shutil
import sys
from functools import partial

import torch
import tqdm
from absl import app, flags
from accelerate.logging import get_logger
from ml_collections import config_flags

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

from sampling.base import generate_dm, generate_dm_concave
from sampling.asyn import generate_asyn
from utils.utils import seed_everything
from utils.sampling import prepare_encoded_prompts
from utils.setup import prepare_accelerator, prepare_pipeline

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/config.py", "Sampling configuration.")

logger = get_logger(__name__)


def main(_):
    # basic setup
    config = FLAGS.config
    print(f'========== seed: {config.seed} ==========')
    if torch.cuda.is_available():
        torch.cuda.set_device(config.dev_id)

    unique_id = config.exp_name if config.exp_name else datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    save_dir = os.path.join(config.save_path, unique_id)

    seed_everything(config.seed)

    accelerator = prepare_accelerator(config, save_dir)
    pipeline = prepare_pipeline(config, accelerator)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
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
    if global_idx:
        with open(os.path.join(save_dir, f'prompt.json'), 'r') as f:
            total_prompts1 = json.load(f)[:global_idx]
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
            cross_mask = generate_dm(config, accelerator, pipeline, idx, prompt_embeds1_combine)

        # ================================================================= #
        # base2 (DM concave)
        if config.generate_dm_concave:
            generate_dm_concave(config, accelerator, pipeline, idx, prompt_embeds1_combine)

        # ================================================================= #
        # asyn
        generate_asyn(config, accelerator, pipeline, idx, prompt_embeds1_combine, cross_mask)

        global_idx += config.sample.batch_size

    with open(os.path.join(save_dir, f'prompt.json'), 'w') as f:
        json.dump(total_prompts1, f)
    shutil.copy("config/config.py", save_dir)


if __name__ == "__main__":
    app.run(main)
