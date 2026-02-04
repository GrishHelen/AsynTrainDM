import os

import torch

from model.unet_2d_condition import unet_asyn_forward


def generate_timesteps_tensor(pipeline, batch_size, type='random'):
    device = pipeline.unet.device
    ts = pipeline.scheduler.timesteps.to(device)  # [T...0]
    res_shape = (batch_size, 64, 64)

    if type == 'random':
        indices = torch.randint(low=0, high=len(ts), size=res_shape, device=device)
        tensor_t = ts[indices]
    else:
        raise NotImplementedError
    return tensor_t.repeat(4, 1, 1, 1).swapaxes(0, 1)


def add_noise(pipeline, clear_latents, ts_tensor=None):
    if ts_tensor is None:
        ts_tensor = generate_timesteps_tensor(pipeline, batch_size=clear_latents.shape[0])

    device = clear_latents.device
    noise = torch.randn_like(clear_latents, device=device)

    alpha_cumprod_t = pipeline.scheduler.alphas_cumprod.to(device)[ts_tensor]
    alpha_cumprod_t = alpha_cumprod_t.expand_as(clear_latents)

    noisy_latent = torch.sqrt(alpha_cumprod_t) * clear_latents + torch.sqrt(1 - alpha_cumprod_t) * noise

    return noisy_latent


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
