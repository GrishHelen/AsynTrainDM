import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers import StableDiffusionPipeline
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.sampling import encode_prompts_list


class DiffusionDBDataset(Dataset):
    def __init__(self, orig_dataset, image_transform, text_transform, load_masks=False):
        self.dataset = orig_dataset
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.load_masks = load_masks

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        prompt = item['prompt']
        mask = item.get('mask', None)
        image = self.image_transform(image)
        prompt = self.text_transform(prompt)
        result = {
            'image': image,
            'prompt_embeds': prompt
        }
        if self.load_masks:
            if mask is None:
                raise ValueError(f"Mask for item {idx} doesn't exist")
            result['mask'] = F.interpolate(torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                           (64, 64)).squeeze()
        return result


def prepare_accelerator(config, save_dir):
    accelerator_config = ProjectConfiguration(
        project_dir=save_dir,
        automatic_checkpoint_naming=True,
        total_limit=100,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.finetune.grad_accumulation_steps
    )

    return accelerator


def prepare_pipeline(config, accelerator, finetuning=False):
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, torch_dtype=inference_dtype)  # float16
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
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
            lora_dropout=config.finetune.lora_dropout,
            # bias="none"
        )
        pipeline.unet.add_adapter(lora_config)
        # pipeline.unet = get_peft_model(pipeline.unet, lora_config)

        # Make sure the trainable params are in float32.
        if accelerator.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(pipeline.unet, dtype=torch.float32)

        print('learnable params:', sum(p.numel() for p in pipeline.unet.parameters() if p.requires_grad))

    else:
        if 'finetuned_model' in config.sample and config.sample.finetuned_model:
            lora_config = LoraConfig(
                r=config.finetune.lora_rank,
                lora_alpha=config.finetune.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
                lora_dropout=config.finetune.lora_dropout,
                # bias="none"
            )
            pipeline.unet.add_adapter(lora_config)
            pipeline.unet.requires_grad_(False)
            pipeline.unet.load_state_dict(torch.load(config.sample.finetuned_model))
            pipeline.unet.eval()

            # for name, param in pipeline.unet.named_parameters():
            #     if "lora_B" in name or "lora_A" in name:
            #         param.data.zero_()

        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    # pipeline.unet.enable_gradient_checkpointing()

    pipeline.vae = accelerator.prepare_model(pipeline.vae, evaluation_mode=True)
    pipeline.text_encoder = accelerator.prepare_model(pipeline.text_encoder, evaluation_mode=True)
    pipeline.unet = accelerator.prepare_model(pipeline.unet, evaluation_mode=not finetuning)

    return pipeline


def prepare_dataloaders(config, pipeline, accelerator):
    device = accelerator.device
    orig_dataset = datasets.load_from_disk(config.finetune.dataset_dir)
    img_size = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    image_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(pipeline.vae.dtype),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    text_transform = lambda prompt: torch.squeeze(encode_prompts_list(pipeline, device, [prompt]))

    dataset = DiffusionDBDataset(orig_dataset, image_transform=image_transform, text_transform=text_transform,
                                 load_masks=config.finetune.use_masks)
    train_loader = DataLoader(dataset=dataset, batch_size=config.finetune.batch_size, shuffle=True, num_workers=0)
    train_loader = accelerator.prepare_data_loader(train_loader)
    return train_loader


def prepare_optimizer(config, pipeline, accelerator):
    if config.finetune.optimizer == 'AdamW':
        params_to_optimize = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=config.finetune.lr,
        )
        optimizer = accelerator.prepare_optimizer(optimizer)
        return optimizer
    if config.finetune.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            pipeline.unet.parameters(), lr=config.finetune.lr
        )
        optimizer = accelerator.prepare_optimizer(optimizer)
        return optimizer

    raise NotImplementedError(f'unknown optimizer {config.finetune.optimizer}')
