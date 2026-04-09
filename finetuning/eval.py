from functools import partial

import torch
import tqdm
from diffusers import DDIMScheduler

from sampling.all import sample_all

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def val_epoch(config, accelerator, pipeline, images_save_dir):
    accelerator.is_local_main_process = False
    autocast = accelerator.autocast
    pipeline.unet.eval()
    with autocast():
        with torch.no_grad():
            old_scheduler = pipeline.scheduler
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

            sample_all(config, accelerator, pipeline, img_save_dir=images_save_dir)

            pipeline.scheduler = old_scheduler
