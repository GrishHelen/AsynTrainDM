import os
from enum import Enum

import torch

from model.unet_2d_condition import unet_asyn_forward


class FinetuneTsType(Enum):
    CONST = 'constant'
    CONST_DELTA = 'constant_delta'
    BLOCK_2X2 = 'block_2x2'
    RANDOM = 'random'


def generate_timesteps_tensor(pipeline, batch_size, type: FinetuneTsType = FinetuneTsType.RANDOM):
    device = pipeline.unet.device
    ts = pipeline.scheduler.timesteps.to(device)  # [T...0]
    res_shape = (batch_size, 64, 64)

    if type == FinetuneTsType.RANDOM:
        indices = torch.randint(low=0, high=len(ts), size=res_shape, device=device)
        tensor_t = ts[indices]
    elif type == FinetuneTsType.CONST:
        t_idx = torch.randint(low=0, high=len(ts), size=(1,), device=device)
        tensor_t = torch.ones(res_shape, device=device) * ts[t_idx]
        tensor_t = tensor_t.to(dtype=ts.dtype)
    elif type == FinetuneTsType.CONST_DELTA:
        t_idx = torch.randint(low=0, high=len(ts), size=(1,), device=device)
        delta = int(0.2 * len(ts))
        deltas = torch.randint(low=-delta, high=delta, size=res_shape, device=device)
        indices = torch.clamp(t_idx + deltas, min=0, max=len(ts) - 1)
        tensor_t = ts[indices]
    else:
        raise NotImplementedError(f'{type.name} is not implemented')
    return tensor_t.repeat(4, 1, 1, 1).swapaxes(0, 1)


def add_noise(scheduler, original_samples, noise, timesteps):
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
    sqrt_alpha_prod = alphas_cumprod ** 0.5
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod) ** 0.5

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    return noisy_samples


def predict_noise(config, pipeline, noisy_latents, timesteps, prompt_embeds1_combine):
    latents_input = torch.cat([noisy_latents] * 2) if config.sample.cfg else noisy_latents
    latents_input = pipeline.scheduler.scale_model_input(latents_input)

    concat_t = torch.cat([timesteps.reshape(-1, 64 * 64)] * 2).round().long()

    noise_pred = unet_asyn_forward(pipeline.unet,
                                   latents_input,
                                   # t,
                                   concat_t,
                                   encoder_hidden_states=prompt_embeds1_combine,
                                   return_dict=False,
                                   extra_input={
                                       'used_layer_size': 16,
                                   },
                                   )
    noise_pred = noise_pred[0]

    if config.sample.cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                noise_pred_text - noise_pred_uncond)

    return noise_pred


def array_to_file(save_dir, file_name, array):
    with open(os.path.join(save_dir, file_name), mode='a') as f:
        f.write('\n'.join(map(str, array + [''])))
