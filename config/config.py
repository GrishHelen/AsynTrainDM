import argparse
import datetime
import json
import os

import ml_collections

from finetuning.utils import FinetuneType, FinetuneTsType
from finetuning.asyndm import FinetuneWarmupType


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
    config.generate_dm_concave = 0
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
    sample.finetuned_model = None
    sample.item_k = None
    sample.save_attn_grids = False
    sample.attn_grid_every = 5
    sample.attn_grid_cell_size = 192
    sample.attn_grid_dir = "attn_grids"
    sample.attn_mask_threshold_type = "mean"  # mean / robust

    ###### Fine-tuning ######
    config.finetune = finetune = ml_collections.ConfigDict()
    finetune.dataset_dir = '/home/ergrishina_2/Diploma/laion'
    finetune.batch_size = 3
    finetune.n_epochs = 50
    finetune.max_batches = -1
    finetune.lora_rank = 32
    finetune.lora_alpha = 64
    finetune.lora_dropout = 0.0
    finetune.max_grad_norm = 1.0
    finetune.grad_accumulation_steps = 1
    finetune.optimizer = 'AdamW'
    finetune.lr = 5e-4
    finetune.ts_type = FinetuneTsType.RANDOM
    finetune.use_masks = False
    finetune.use_ltg = False
    finetune.mask_source = "dataset"  # dataset / attention
    finetune.item_idx_file = ""
    finetune.attn_mask_threshold_type = "robust"  # mean / robust
    # Kept for compatibility with existing configs; attention-mask training now fuses 16x16 and 32x32 maps.
    finetune.attn_mask_used_layer_size = 16
    finetune.type = FinetuneType.Asyn
    finetune.item_k = 0.7

    finetune.ltg = ml_collections.ConfigDict()
    finetune.ltg.loc = 0.5
    finetune.ltg.scale = 1.0
    finetune.ltg.std = 0.6
    finetune.ltg.block_size = 1

    finetune.schedule_warmup = ml_collections.ConfigDict()
    finetune.schedule_warmup.n_epochs = 0
    finetune.schedule_warmup.type = FinetuneWarmupType.POLYNOM # polynom / mixture

    ###### Logging ######
    config.logging = logging = ml_collections.ConfigDict()
    logging.eval_epoch = 2
    logging.metrics_epoch = 0  # 0 disables metric evaluation
    logging.metrics_datasets = ["animal", "drawbench"]
    logging.metrics_samples_per_prompt = 1
    logging.metrics_clip_model = "openai/clip-vit-large-patch14"
    logging.metrics_qwen_model = "Qwen/Qwen2.5-VL-7B-Instruct"
    logging.metrics_device = "auto"

    ###### Heatmap Parameters ######
    config.heatmap = heatmap = ml_collections.ConfigDict()
    # visualize heatmaps for every k timesteps
    heatmap.every_k = 5

    return config


def configure_metrics_logging(config, args):
    defaults = get_default_config().logging
    if "logging" not in config:
        config.logging = defaults
    for key in ("metrics_epoch", "metrics_datasets", "metrics_samples_per_prompt",
                "metrics_clip_model", "metrics_qwen_model", "metrics_device"):
        value = getattr(args, key)
        if value is not None:
            config.logging[key] = value
        elif key not in config.logging:
            config.logging[key] = defaults[key]

    if config.logging.metrics_epoch < 0:
        raise ValueError("metrics_epoch must be nonnegative (0 disables metric evaluation)")
    from metrics.gen_images import get_samples_per_prompt

    get_samples_per_prompt(config)
    if config.logging.metrics_epoch:
        from finetuning.metrics import get_metrics_datasets

        config.logging.metrics_datasets = get_metrics_datasets(config)


def get_config():
    parser = argparse.ArgumentParser(description="Parsing arguments for config from console")

    # ready config
    parser.add_argument("--config_path", type=str, default=None)

    # config args
    parser.add_argument("--mixed_precision", "--mp", type=str, default="no")
    parser.add_argument("--exp_name", "--exp", "--name", type=str, default=None)
    parser.add_argument("--generate_dm_concave", "--dm_concave", type=int, default=0)
    parser.add_argument("--generate_dm", "--dm", type=int, default=1)
    parser.add_argument("--prompt_file", type=str, default="")
    parser.add_argument("--items_file", type=str, default="")
    parser.add_argument("--mask_thr", type=float, default=None)

    # config.pretrained args
    parser.add_argument("--pretrained_model", "--pretrained", type=str,
                        default="/home/ergrishina_2/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/repo/")

    # config.sample args
    parser.add_argument("--sample_batch_size", "--sample_bs", type=int, default=4)
    parser.add_argument("--finetuned_model", "--finetuned", type=str, default=None)
    parser.add_argument("--sample_k", type=float, default=None)
    parser.add_argument("--save_attn_grids", "--save_cross_attention_grids", type=int, default=0)
    parser.add_argument("--attn_grid_every", "--cross_attention_grid_every", type=int, default=5)
    parser.add_argument("--attn_grid_cell_size", type=int, default=192)
    parser.add_argument(
        "--sample_attn_mask_threshold_type",
        choices=["mean", "robust"],
        default="mean",
    )

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
    parser.add_argument("--finetune_use_ltg", "--use_ltg", type=int, default=0)
    parser.add_argument("--finetune_mask_source", type=str, default="dataset")
    parser.add_argument("--finetune_items_file", "--finetune_item_idx_file", type=str, default="")
    parser.add_argument(
        "--finetune_attn_mask_threshold_type",
        "--attn_mask_threshold_type",
        choices=["mean", "robust"],
        default="mean",
    )
    parser.add_argument("--finetune_attn_mask_used_layer_size", "--attn_mask_used_layer_size", type=int, default=16)
    parser.add_argument("--finetune_ltg_loc", "--ltg_loc", type=float, default=0.5)
    parser.add_argument("--finetune_ltg_scale", "--ltg_scale", type=float, default=1.0)
    parser.add_argument("--finetune_ltg_std", "--ltg_std", type=float, default=0.6)
    parser.add_argument("--finetune_ltg_block_size", "--ltg_block_size", type=int, default=1)
    parser.add_argument("--finetune_type", type=str, default='asyn')
    parser.add_argument("--finetune_item_k", type=float, default=0.7)
    
    parser.add_argument("--finetune_warmup_epochs", type=int, default=0)
    parser.add_argument("--finetune_warmup_type", type=str, default='polynom')

    # config.logging args
    parser.add_argument("--log_epoch", type=int, default=5)
    parser.add_argument("--eval_epoch", type=int, default=2)
    parser.add_argument("--metrics_epoch", type=int, default=None,
                        help="Compute CLIP and Qwen after every N completed epochs; 0 disables (default)")
    parser.add_argument("--metrics_datasets", nargs="+", default=None,
                        help="Validation prompt sets from config/prompt, e.g. animal drawbench coyo_test")
    parser.add_argument("--metrics_samples_per_prompt", "--samples_per_prompt", type=int, default=None,
                        help="Images per prompt for metrics, generated in successive full passes (default: 1)")
    parser.add_argument("--metrics_clip_model", type=str, default=None,
                        help="CLIP model ID or local model directory")
    parser.add_argument("--metrics_qwen_model", type=str, default=None,
                        help="Qwen model ID or local model directory")
    parser.add_argument("--metrics_device", type=str, default=None,
                        help="Device for metric models: auto (training device), cpu, cuda:0, ...")

    args = parser.parse_args()
    if args.config_path is not None:
        with open(args.config_path, "r", encoding="utf-8") as f:
            loaded_dict = json.load(f)
        loaded_config = ml_collections.ConfigDict(loaded_dict)
        configure_metrics_logging(loaded_config, args)
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
    config.pretrained.model = args.pretrained_model

    # config.sample args
    config.sample.finetuned_model = args.finetuned_model
    config.sample.batch_size = args.sample_batch_size
    if args.sample_k is not None and not 0.0 <= args.sample_k <= 1.0:
        raise ValueError(f"sample_k must be in [0, 1], got {args.sample_k}")
    config.sample.item_k = args.sample_k
    config.sample.save_attn_grids = bool(args.save_attn_grids)
    config.sample.attn_grid_every = args.attn_grid_every
    config.sample.attn_grid_cell_size = args.attn_grid_cell_size
    config.sample.attn_mask_threshold_type = args.sample_attn_mask_threshold_type
    config.prompt_file = args.prompt_file
    config.item_idx_file = args.items_file
    if args.mask_thr is not None:
        config.mask_thr = args.mask_thr

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
    config.finetune.use_ltg = bool(args.finetune_use_ltg)
    config.finetune.mask_source = args.finetune_mask_source.lower()
    if config.finetune.mask_source not in ["dataset", "attention"]:
        raise ValueError(f"Unknown finetune_mask_source: {args.finetune_mask_source}")
    config.finetune.item_idx_file = args.finetune_items_file
    config.finetune.attn_mask_threshold_type = args.finetune_attn_mask_threshold_type
    config.finetune.attn_mask_used_layer_size = args.finetune_attn_mask_used_layer_size
    config.finetune.ltg.loc = args.finetune_ltg_loc
    config.finetune.ltg.scale = args.finetune_ltg_scale
    config.finetune.ltg.std = args.finetune_ltg_std
    config.finetune.ltg.block_size = args.finetune_ltg_block_size
    if args.finetune_type == 'asyn':
        config.finetune.type = FinetuneType.Asyn
    elif args.finetune_type == 'asyndm':
        config.finetune.type = FinetuneType.AsynDM
    config.finetune.item_k = args.finetune_item_k
    
    config.finetune.schedule_warmup.n_epochs = args.finetune_warmup_epochs
    if args.finetune_warmup_type == 'polynom':
        config.finetune.schedule_warmup.type = FinetuneWarmupType.POLYNOM
    elif args.finetune_warmup_type == 'mixture':
        config.finetune.schedule_warmup.type = FinetuneWarmupType.MIXTURE
    else:
        raise ValueError(f'Unknown schedule_warmup type: {args.finetune_warmup_type}')

    # config.logging args
    config.logging.eval_epoch = args.eval_epoch
    configure_metrics_logging(config, args)

    save_config(config)
    return config
