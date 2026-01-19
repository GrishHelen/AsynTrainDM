import datasets
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader


def prepare_accelerator(config, save_dir):
    accelerator_config = ProjectConfiguration(
        project_dir=save_dir,
        automatic_checkpoint_naming=True,
        total_limit=100,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config
    )

    return accelerator


def prepare_pipeline(config, accelerator, finetuning=False):
    # load
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.float16)  # float16
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(finetuning)
    # disable safety checker
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    if finetuning:
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    else:
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    return pipeline


def prepare_dataloaders(config):
    dataset = datasets.load_from_disk(config.finetune.dataset_dir)

    if config.finetune.val_size:
        train_dataset, val_dataset = dataset.train_test_split(test_size=config.finetune.val_size, seed=42)
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.finetune.batch_size, shuffle=True,
                                  num_workers=-1)
        val_loader = DataLoader(dataset=val_dataset, batch_size=config.finetune.batch_size, shuffle=False,
                                num_workers=-1)
        return train_loader, val_loader

    train_loader = DataLoader(dataset=dataset, batch_size=config.finetune.batch_size, shuffle=True, num_workers=-1)
    return train_loader, None
