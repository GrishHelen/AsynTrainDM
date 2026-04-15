import datetime
import os
import sys
from functools import partial

import torch
import tqdm
from accelerate.logging import get_logger

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

from config.config import get_config
from sampling.all import sample_all
from utils.utils import seed_everything
from utils.setup import prepare_accelerator, prepare_pipeline

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

logger = get_logger(__name__)


def main(config):
    print(f'========== seed: {config.seed} ==========')
    if torch.cuda.is_available():
        torch.cuda.set_device(config.dev_id)

    unique_id = config.exp_name if config.exp_name else datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    print(f'Sampling, experiment: {unique_id}')
    save_dir = os.path.join(config.save_path, unique_id)

    seed_everything(config.seed)

    accelerator = prepare_accelerator(config, save_dir)
    pipeline = prepare_pipeline(config, accelerator)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    sample_all(config, accelerator, pipeline, save_dir=save_dir)


if __name__ == "__main__":
    config = get_config()
    main(config)
