import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # random seed for reproducibility.
    config.seed = 1234
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "fp16"  # "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # sample path
    config.save_path = "../results"
    # exp name
    config.exp_name = "AsynDM, finetune"
    # gpu id
    config.dev_id = 0
    # prompt directly used
    config.prompt = [
        "a rabbit playing basketball",
        # "a white car and a red sheep",
        # "a cartoon style illustration of a macaw skating",
        # "a photo of a gift box at the fireplace",
        # "a photo of a cute ostrich on the chair",
        # "a photo of a cute ostrich in the cave",
        # "a photo of an orange candle in the bathroom",
        # "a photo of an orange candle among the snowdrifts",
        # "a painting of an orange on the roof",
    ]
    # prompt file
    config.prompt_file = ""
    # cross mask threshold
    config.mask_thr = 1.0
    # item idx in prompt
    config.item_idx = [
        [1, 3],
        # [2, 6],
        # [6],
        # [5, 8],
        # [5, 8],
        # [5, 8],
        # [5, 8],
        # [5, 8],
        # [4, 7],
    ]  # [1,5,11][1,7,13][3,9][1,4][2,4]
    # item k in prompt
    config.item_k = [
        [0.7, 0.7],  # [0.7, 0.7], [0.7],
        # [0.7, 0.7], [0.7, 0.7], [0.7, 0.7],
        # [0.7, 0.7], [0.7, 0.7], [0.7, 0.7], [0.7, 0.7]
    ]
    # use static or dynamic mask
    config.static_mask = 0
    # item idx file
    config.item_idx_file = ""
    # whether generate base2 (DM concave)
    config.generate_dm_concave = 0
    # whether generate base (DM)
    config.generate_dm = 0
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
    # whether use classifier-free guidance
    sample.cfg = True

    ###### Fine-tuning ######
    config.finetune = finetune = ml_collections.ConfigDict()
    finetune.dataset_dir = '../diffusiondb'
    finetune.batch_size = 32
    finetune.val_size = 0.2
    finetune.n_epochs = 3

    ###### Heatmap Parameters ######
    config.heatmap = heatmap = ml_collections.ConfigDict()
    # visualize heatmaps for every k timesteps
    heatmap.every_k = 5

    return config
