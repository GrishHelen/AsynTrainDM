import os
from enum import Enum

import torch

from model.unet_2d_condition import unet_asyn_forward


class FinetuneTsType(Enum):
    CONST = 'constant'
    CONST_DELTA = 'constant_delta'
    BLOCK_2X2 = 'block_2x2'
    RANDOM = 'random'


class FinetuneType(Enum):
    Asyn = 'asyn'
    AsynDM = 'asyndm'


def generate_timesteps_tensor(pipeline, batch_size, type: FinetuneTsType = FinetuneTsType.RANDOM):
    device = pipeline.unet.device
    ts = pipeline.scheduler.timesteps.to(device)  # [T-1...0]
    num_train_timesteps = pipeline.scheduler.config.num_train_timesteps
    res_shape = (batch_size, 64, 64)

    if type == FinetuneTsType.RANDOM:
        tensor_t = torch.randint(low=0, high=num_train_timesteps, size=res_shape, device=device)
    elif type == FinetuneTsType.CONST:
        t = torch.randint(low=0, high=num_train_timesteps, size=(1,), device=device)
        tensor_t = torch.ones(res_shape, device=device, dtype=t.dtype) * t
    elif type == FinetuneTsType.CONST_DELTA:
        t = torch.randint(low=0, high=num_train_timesteps, size=(1,), device=device)
        delta = int(0.2 * num_train_timesteps)
        deltas = torch.randint(low=-delta, high=delta, size=res_shape, device=device)
        tensor_t = torch.clamp(t + deltas, min=0, max=num_train_timesteps - 1)
    else:
        raise NotImplementedError(f'{type.name} is not implemented')
    return tensor_t.long()


def generate_ltg_timesteps_tensor(config, pipeline, batch_size, height=64, width=64):
    device = pipeline.unet.device
    ltg_config = config.finetune.get("ltg", {})
    loc = ltg_config.get("loc", 0.5)
    scale = ltg_config.get("scale", 1.0)
    std = ltg_config.get("std", 0.6)
    block_size = int(ltg_config.get("block_size", 1))
    if block_size <= 0:
        raise ValueError(f"ltg.block_size must be positive, got {block_size}")
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(f"ltg.block_size={block_size} must divide timestep grid {(height, width)}")

    grid_h = height // block_size
    grid_w = width // block_size
    num_patches = grid_h * grid_w

    clean_t_max = torch.sigmoid(loc + scale * torch.randn(batch_size, device=device, dtype=torch.float32))
    std_eff = torch.minimum(clean_t_max / 2, torch.full_like(clean_t_max, std))
    eps = torch.randn(batch_size, num_patches, device=device, dtype=torch.float32)
    clean_t = clean_t_max[:, None] - eps.abs() * std_eff[:, None]

    fallback = torch.rand_like(clean_t) * clean_t_max[:, None]
    clean_t = torch.where(clean_t < 0, fallback, clean_t)
    clean_t = clean_t.view(batch_size, grid_h, grid_w)
    if block_size > 1:
        clean_t = clean_t.repeat_interleave(block_size, dim=1).repeat_interleave(block_size, dim=2)

    num_train_timesteps = pipeline.scheduler.config.num_train_timesteps
    tensor_t = ((1 - clean_t) * (num_train_timesteps - 1)).round().long()
    return torch.clamp(tensor_t, min=0, max=num_train_timesteps - 1)


def add_noise(scheduler, original_samples, noise, timesteps):
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
    if len(alphas_cumprod.shape) < len(original_samples.shape):
        alphas_cumprod = alphas_cumprod.unsqueeze(1)

    sqrt_alpha_prod = alphas_cumprod ** 0.5
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod) ** 0.5

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    return noisy_samples


def predict_noise(config, pipeline, noisy_latents, timesteps, prompt_embeds):
    latents_input = noisy_latents
    latents_input = pipeline.scheduler.scale_model_input(latents_input)

    concat_t = timesteps.reshape(timesteps.shape[0], -1).round().long()

    noise_pred = unet_asyn_forward(pipeline.unet,
                                   latents_input,
                                   # t,
                                   concat_t,
                                   encoder_hidden_states=prompt_embeds,
                                   return_dict=False,
                                   extra_input={
                                       'used_layer_size': 16,
                                   },
                                   )
    noise_pred = noise_pred[0]

    return noise_pred


def array_to_file(save_dir, file_name, array):
    with open(os.path.join(save_dir, file_name), mode='a') as f:
        f.write('\n'.join(map(str, array + [''])))
