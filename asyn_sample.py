import contextlib
import os
import datetime
import time
import sys
import shutil
import torch
import torch.nn.functional as F
from functools import partial
import tqdm
from PIL import Image
import json
import random
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import DDIMScheduler
import numpy as np

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

from sampling.base import generate_dm, generate_dm_concave
from sampling.asyn import generate_asyn
from utils.utils import seed_everything

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

    accelerator_config = ProjectConfiguration(
        project_dir=save_dir,
        automatic_checkpoint_naming=True,
        total_limit=100,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config
    )

    # load
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.float16)  # float16
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    # disable safety checker
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    total_image_num_per_gpu = config.sample.batch_size * config.sample.num_batches_per_epoch
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)

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
        prompts1 = [
            prompt_list[prompt_idx]
            for _ in range(config.sample.batch_size)
        ]
        total_prompts1.extend(prompts1)
        # encode prompts
        prompt_ids1 = pipeline.tokenizer(
            prompts1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]
        # combine prompt and neg_prompt
        prompt_embeds1_combine = torch.cat([sample_neg_prompt_embeds, prompt_embeds1], dim=0)
        cross_mask = None

        # ================================================================= #
        # base (DM)
        if config.generate_dm or config.static_mask:
            cross_mask = generate_dm(config, accelerator, pipeline, idx, prompt_list, prompt_embeds1_combine)

        # ================================================================= #
        # base2 (DM concave)
        if config.generate_dm_concave:
            generate_dm_concave(config, accelerator, pipeline, idx, prompt_list, prompt_embeds1_combine)

        # ================================================================= #
        # asyn
        generate_asyn(config, accelerator, pipeline, idx, prompt_list, prompt_embeds1_combine, cross_mask)

        global_idx += config.sample.batch_size

    with open(os.path.join(save_dir, f'prompt.json'), 'w') as f:
        json.dump(total_prompts1, f)
    shutil.copy("config/config.py", save_dir)


if __name__ == "__main__":
    app.run(main)
