import datasets
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from utils.sampling import encode_prompts_list


class DiffusionDBDataset(Dataset):
    def __init__(self, orig_dataset, image_transform, text_transform):
        self.dataset = orig_dataset
        self.image_transform = image_transform
        self.text_transform = text_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        prompt = item['prompt']
        image = self.image_transform(image)
        prompt = self.text_transform(prompt)
        result = {
            'image': image,
            'prompt_embeds': prompt
        }
        return result


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
    if finetuning:
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

        # LoRA
        lora_config = LoraConfig(
            r=config.finetune.lora_rank,
            lora_alpha=config.finetune.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=config.finetune.lora_dropout,
            bias="none"
        )
        pipeline.unet = get_peft_model(pipeline.unet, lora_config)

        print('learnable params:', sum(p.numel() for p in pipeline.unet.parameters() if p.requires_grad))

    else:
        if 'finetuned_model' in config.sample:
            lora_config = LoraConfig(
                r=config.finetune.lora_rank,
                lora_alpha=config.finetune.lora_alpha,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                lora_dropout=config.finetune.lora_dropout,
                bias="none"
            )
            pipeline.unet = get_peft_model(pipeline.unet, lora_config)
            pipeline.unet.requires_grad_(False)
            pipeline.unet.load_state_dict(torch.load(config.sample.finetuned_model))
            pipeline.unet.eval()

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
    # pipeline.unet.enable_gradient_checkpointing()

    return pipeline


def prepare_dataloaders(config, pipeline, device):
    orig_dataset = datasets.load_from_disk(config.finetune.dataset_dir)
    img_size = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float16),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    text_transform = lambda prompt: torch.squeeze(encode_prompts_list(pipeline, device, [prompt]))

    if config.finetune.val_ratio:
        train_subset, val_subset = random_split(orig_dataset,
                                                lengths=[1 - config.finetune.val_ratio, config.finetune.val_ratio])
        train_dataset = DiffusionDBDataset(train_subset, image_transform=image_transform, text_transform=text_transform)
        val_dataset = DiffusionDBDataset(val_subset, image_transform=image_transform, text_transform=text_transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.finetune.batch_size, shuffle=True,
                                  num_workers=0)
        val_loader = DataLoader(dataset=val_dataset, batch_size=config.finetune.batch_size, shuffle=False,
                                num_workers=0)
        return train_loader, val_loader

    dataset = DiffusionDBDataset(orig_dataset, image_transform=image_transform, text_transform=text_transform)
    train_loader = DataLoader(dataset=dataset, batch_size=config.finetune.batch_size, shuffle=True, num_workers=0)
    return train_loader, None


def prepare_optimizer(config, pipeline):
    if config.finetune.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            pipeline.unet.parameters(), lr=config.finetune.lr, fused=True
        )
        return optimizer
    if config.finetune.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            pipeline.unet.parameters(), lr=config.finetune.lr
        )
        return optimizer

    raise NotImplementedError(f'unknown optimizer {config.finetune.optimizer}')
