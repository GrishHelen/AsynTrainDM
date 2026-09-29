from functools import partial

import torch
import tqdm
from diffusers import DDIMScheduler

from sampling.all import sample_all

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def val_epoch(config, accelerator, pipeline, images_save_dir):
    autocast = accelerator.autocast
    pipeline.unet.eval()
    with autocast():
        with torch.no_grad():
            old_scheduler = pipeline.scheduler
            old_save_attn_grids = config.sample.get("save_attn_grids", False)
            old_attn_grid_dir = config.sample.get("attn_grid_dir", "attn_grids")

            save_validation_attn_grids = config.finetune.get("mask_source", "dataset") == "attention"
            if save_validation_attn_grids:
                config.sample.save_attn_grids = True
                config.sample.attn_grid_dir = "attn_grid"

            try:
                pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
                sample_all(config, accelerator, pipeline, img_save_dir=images_save_dir)
            finally:
                pipeline.scheduler = old_scheduler
                config.sample.save_attn_grids = old_save_attn_grids
                config.sample.attn_grid_dir = old_attn_grid_dir
