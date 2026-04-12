import argparse
import datetime
import json
import os

import ml_collections

from finetuning.utils import FinetuneType, FinetuneTsType


def save_config(config):
    unique_id = config.exp_name if config.exp_name else datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    save_dir = os.path.join(config.save_path, unique_id)
    save_path = os.path.join(save_dir, 'config.json')
    os.makedirs(save_dir, exist_ok=True)

    json_data = config.to_json_best_effort(indent=2)
    with open(save_path, 'w') as f:
        f.write(json_data)


def get_default_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # random seed for reproducibility.
    config.seed = 1234
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "no"  # "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # sample path
    config.save_path = "../results"
    # exp name
    config.exp_name = "AsynDM, finetuned_3"
    # gpu id
    config.dev_id = 0
    # prompt directly used
    config.prompt = [
        "a rabbit playing basketball",
        "a white car and a red sheep",
        # "a cartoon style illustration of a macaw skating",

        "a cute ostrich on the chair",
        # "a penguin wearing a straw hat",
        "a blue cat and a gray rabbit",
    ]
    # prompt file
    config.prompt_file = ""
    # cross mask threshold
    config.mask_thr = 1.0
    # item idx in prompt
    config.item_idx = [
        [1, 3],
        [2, 6],
        # [6],

        [2, 5],
        # [1, 5],
        [2, 6],
    ]  # [1,5,11][1,7,13][3,9][1,4][2,4]
    # item k in prompt
    config.item_k = [
        [0.7, 0.7], [0.7, 0.7],  # [0.7],

        [0.7, 0.7], [0.7, 0.7],  # [0.7, 0.7],
    ]
    # use static or dynamic mask
    config.static_mask = 0
    # item idx file
    config.item_idx_file = ""
    # whether generate base2 (DM concave)
    config.generate_dm_concave = 1
    # whether generate base (DM)
    config.generate_dm = 1
    # batch begin index
    config.begin_index = 0
    # curve type
    config.curve_type = "bin"  # "bin", "lin", "exp"

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "/home/ergrishina_2/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/repo/"  # "stabilityai/stable-diffusion-2-1" or "path/to/your/sd2.1-base"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 50
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 5.0
    # batch size (per GPU!) to use for sampling.
    sample.batch_size = 4
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample.num_batches_per_epoch = 1
    # whether to use classifier-free guidance
    sample.cfg = True
    sample.finetuned_model = "/home/ergrishina_2/Diploma/results/AsynDM, finetune_3/models_state_dict/model__10.pth"

    ###### Fine-tuning ######
    config.finetune = finetune = ml_collections.ConfigDict()
    finetune.dataset_dir = '/home/ergrishina_2/Diploma/diffusiondb'
    finetune.batch_size = 3
    finetune.n_epochs = 40
    finetune.max_batches = -1
    finetune.lora_rank = 4
    finetune.lora_alpha = 8
    finetune.lora_dropout = 0.0
    finetune.max_grad_norm = 1.0
    finetune.grad_accumulation_steps = 1
    finetune.optimizer = 'AdamW'
    finetune.lr = 5e-4
    finetune.ts_type = FinetuneTsType.RANDOM
    finetune.use_masks = False
    finetune.type = FinetuneType.Asyn

    ###### Logging ######
    config.logging = logging = ml_collections.ConfigDict()
    logging.batch = 50
    logging.epoch = 5
    logging.eval_epoch = 2

    ###### Heatmap Parameters ######
    config.heatmap = heatmap = ml_collections.ConfigDict()
    # visualize heatmaps for every k timesteps
    heatmap.every_k = 5

    return config


def get_config():
    parser = argparse.ArgumentParser(description="Parsing arguments for config from console")

    # ready config
    parser.add_argument("--config_path", type=str, default=None)

    # config args
    parser.add_argument("--mixed_precision", "--mp", type=str, default="no")
    parser.add_argument("--exp_name", "--exp", "--name", type=str, default=None)
    parser.add_argument("--generate_dm_concave", "--dm_concave", type=int, default=0)
    parser.add_argument("--generate_dm", "--dm", type=int, default=1)

    # config.pretrained args
    parser.add_argument("--pretrained_model", "--pretrained", type=str,
                        default="/home/ergrishina_2/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/repo/")

    # config.sample args
    parser.add_argument("--sample_batch_size", "--sample_bs", type=int, default=4)
    parser.add_argument("--finetuned_model", "--finetuned", type=str, default=None)

    # config.finetune args
    parser.add_argument("--finetune_dataset_dir", "--dataset_dir", "--dataset", type=str,
                        default="/home/ergrishina_2/Diploma/laion")
    parser.add_argument("--finetune_batch_size", "--finetune_bs", type=int, default=3)
    parser.add_argument("--finetune_n_epochs", "--finetune_epochs", type=int, default=50)
    parser.add_argument("--finetune_max_batches", "--finetune_batches", type=int, default=-1)
    parser.add_argument("--finetune_lora_rank", type=int, default=32)
    parser.add_argument("--finetune_lora_alpha", type=int, default=None)
    parser.add_argument("--finetune_lora_dropout", type=float, default=0.0)
    parser.add_argument("--finetune_grad_accumulation_steps", "--finetune_acc_steps", type=int, default=1)
    parser.add_argument("--finetune_lr", type=float, default=None)
    parser.add_argument("--finetune_ts_type", type=str, default=None)
    parser.add_argument("--finetune_use_mask", type=int, default=0)
    parser.add_argument("--finetune_type", type=str, default='asyn')

    # config.logging args
    parser.add_argument("--log_epoch", type=int, default=5)
    parser.add_argument("--eval_epoch", type=int, default=2)

    args = parser.parse_args()
    if args.config_path is not None:
        with open("config.json", "r") as f:
            loaded_dict = json.load(f)
        loaded_config = ml_collections.ConfigDict(loaded_dict)
        save_config(loaded_config)
        return loaded_config

    config = get_default_config()

    # config args
    if args.mixed_precision:
        config.mixed_precision = args.mixed_precision
    config.exp_name = args.exp_name if args.exp_name else datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.generate_dm_concave = args.generate_dm_concave
    config.generate_dm = args.generate_dm

    # config.pretrained args
    config.pretrained.pretrained_model = args.pretrained_model

    # config.sample args
    config.sample.finetuned_model = args.finetuned_model
    config.sample.batch_size = args.sample_batch_size

    # config.finetune args
    config.finetune.dataset_dir = args.finetune_dataset_dir
    config.finetune.batch_size = args.finetune_batch_size
    config.finetune.n_epochs = args.finetune_n_epochs
    config.finetune.max_batches = args.finetune_max_batches
    config.finetune.lora_rank = args.finetune_lora_rank
    if args.finetune_lora_alpha is None:
        config.finetune.lora_alpha = 2 * config.finetune.lora_rank
    else:
        config.finetune.lora_alpha = args.finetune_lora_alpha
    config.finetune.lora_dropout = args.finetune_lora_dropout
    config.finetune.grad_accumulation_steps = args.finetune_grad_accumulation_steps
    config.finetune.lr = args.finetune_lr
    if args.finetune_ts_type is None or args.finetune_ts_type in ['const', 'constant']:
        config.finetune.ts_type = FinetuneTsType.CONST
    elif args.finetune_ts_type in ['const_delta', 'constant_delta']:
        config.finetune.ts_type = FinetuneTsType.CONST_DELTA
    elif args.finetune_ts_type in ['block_2x2', 'block_2']:
        config.finetune.ts_type = FinetuneTsType.BLOCK_2X2
    elif args.finetune_ts_type in ['rand', 'random']:
        config.finetune.ts_type = FinetuneTsType.RANDOM
    else:
        raise ValueError('')
    config.finetune.use_masks = bool(args.finetune_use_mask)
    if args.finetune_type == 'asyn':
        config.finetune.type = FinetuneType.Asyn
    elif args.finetune_type == 'asyndm':
        config.finetune.type = FinetuneType.AsynDM

    # config.logging args
    config.logging.epoch = args.log_epoch
    config.logging.eval_epoch = args.eval_epoch

    save_config(config)
    return config
