import datetime
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

from utils.utils import seed_everything
from utils.setup import prepare_accelerator, prepare_pipeline, prepare_dataloaders, prepare_optimizer
from finetuning.asyn import train_asyn
from utils.sampling import encode_prompts_list

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
    pipeline = prepare_pipeline(config, accelerator, finetuning=True)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # generate negative prompt embeddings
    neg_prompt_embed = encode_prompts_list(pipeline, accelerator.device, [""])
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.finetune.batch_size, 1, 1)

    train_dataloader, val_dataloader = prepare_dataloaders(config, pipeline, accelerator.device)
    optimizer = prepare_optimizer(config, pipeline)

    # asyn
    train_asyn(config, accelerator, pipeline, optimizer, save_dir, train_dataloader, val_dataloader,
               sample_neg_prompt_embeds=sample_neg_prompt_embeds)


if __name__ == "__main__":
    app.run(main)
