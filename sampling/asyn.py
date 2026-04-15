import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image

from diffusion.asyn_ddim_with_logprob import asyn_ddim_step_with_logprob, latents_decode
from model.unet_2d_condition import unet_asyn_forward
from .utils import get_item_idx_list, get_item_k_list, func_prev_linear, func_prev_binary

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def generate_asyn(config, accelerator, pipeline, idx, prompt_embeds1_combine, cross_mask=None, img_save_dir=None):
    global_idx = idx * config.sample.batch_size
    autocast = accelerator.autocast
    prompt_idx = idx // config.sample.num_batches_per_epoch
    if img_save_dir is None:
        img_save_dir = os.path.join(accelerator.project_configuration.project_dir, "images/")
    item_idx_list = get_item_idx_list(config, prompt_idx)
    item_k_list = get_item_k_list(config, prompt_idx)

    gs = [torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu') for _ in range(config.sample.batch_size)]
    for i, g in enumerate(gs):
        g.manual_seed(config.seed + (idx % config.sample.num_batches_per_epoch) * config.sample.batch_size + i)
    noise_latents1 = pipeline.prepare_latents(
        config.sample.batch_size,
        pipeline.unet.config.in_channels,  ## channels
        pipeline.unet.config.sample_size * pipeline.vae_scale_factor,  ## height
        pipeline.unet.config.sample_size * pipeline.vae_scale_factor,  ## width
        prompt_embeds1_combine.dtype,
        accelerator.device,
        gs  ## generator
    )

    item_cnt = len(item_idx_list)
    if not config.static_mask:
        cross_mask = torch.zeros(config.sample.batch_size, item_cnt, 64, 64, dtype=torch.float32,
                                 device=accelerator.device)
        cross_mask[:, np.array(item_idx_list).argmax()] = 1
    bg_mask = 1 - (cross_mask > 0.5).any(dim=1).float()
    initial_t = pipeline.scheduler.config.num_train_timesteps + pipeline.scheduler.config.steps_offset
    initial_t = torch.tensor(initial_t, device=accelerator.device, dtype=torch.float32)
    state_t = initial_t[None, None, None].expand(config.sample.batch_size, 64, 64)
    state_prev_t_linear = func_prev_linear(pipeline, state_t, config.sample.num_steps)
    state_prev_t_binary = []
    for j in range(item_cnt):
        state_prev_t_binary.append(
            cross_mask[:, j] * func_prev_binary(config, pipeline,
                                                state_t, config.sample.num_steps, k=item_k_list[j]))
    state_prev_t_binary = torch.stack(state_prev_t_binary, dim=1).sum(dim=1)
    state_t = (bg_mask * state_prev_t_linear + state_prev_t_binary)
    # print(state_t)

    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(gs, config.sample.eta)

    latents_t = noise_latents1

    for i in tqdm(
            range(config.sample.num_steps),
            desc="Timestep",
            position=3,
            leave=False,
            disable=True,
    ):
        # sample

        with autocast():
            with torch.no_grad():
                latents_input = torch.cat([latents_t] * 2) if config.sample.cfg else latents_t
                latents_input = pipeline.scheduler.scale_model_input(latents_input)

                # print(state_t)
                concat_t = torch.cat([state_t.reshape(-1, 64 * 64)] * 2).round().long()

                bg_mask = 1 - (cross_mask > 0.5).any(dim=1).float()
                state_prev_t_linear = func_prev_linear(pipeline, state_t, config.sample.num_steps - i - 1)
                state_prev_t_binary = []
                for j in range(item_cnt):
                    state_prev_t_binary.append(
                        cross_mask[:, j] * func_prev_binary(config, pipeline,
                                                            state_t, config.sample.num_steps - i - 1,
                                                            k=item_k_list[j]))
                state_prev_t_binary = torch.stack(state_prev_t_binary, dim=1).sum(dim=1)
                state_prev_t = (bg_mask * state_prev_t_linear + state_prev_t_binary)

                tensor_t = state_t[:, None].expand(config.sample.batch_size, 4, 64, 64).round().long()
                tensor_prev_t = state_prev_t[:, None].expand(config.sample.batch_size, 4, 64, 64).round().long()

                noise_pred, extra_inf = unet_asyn_forward(pipeline.unet,
                                                          latents_input,
                                                          # t,
                                                          concat_t,
                                                          encoder_hidden_states=prompt_embeds1_combine,
                                                          return_dict=False,
                                                          extra_input={
                                                              'used_layer_size': 16,
                                                              'item_idx': item_idx_list
                                                          },
                                                          return_extra_inf=True,
                                                          )
                noise_pred = noise_pred[0]
                if config.sample.cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                            noise_pred_text - noise_pred_uncond)
                latents_t_1, _, latents_0 = asyn_ddim_step_with_logprob(pipeline.scheduler,
                                                                        noise_pred,
                                                                        tensor_t,
                                                                        tensor_prev_t,
                                                                        latents_t,
                                                                        **extra_step_kwargs)
                latents_t = latents_t_1

                if not config.static_mask:
                    cross_mask = extra_inf['cross_mask']  # (bsize, width_height, item_idx)
                    mask_mean = config.mask_thr * cross_mask.mean(dim=1, keepdim=True)
                    cross_mask[cross_mask >= mask_mean] = 1
                    cross_mask[cross_mask < mask_mean] = 0

                    bsize, width_height, item_cnt = cross_mask.shape
                    width = int(width_height ** 0.5)
                    cross_mask = cross_mask.permute(0, 2, 1).reshape(bsize, item_cnt, width, width)
                    a_tensor = torch.tensor(item_k_list, dtype=torch.float32,
                                            device=cross_mask.device)  # shape: (item_cnt,)
                    a_tensor = a_tensor.view(1, item_cnt, 1, 1)  # shape: (1, item_cnt, 1, 1)
                    priority_masks = cross_mask * a_tensor  # (bsize, item_cnt, width, width)
                    _, max_idx = priority_masks.max(dim=1)  # shape: (bsize, width, width)
                    final_masks = torch.zeros_like(cross_mask)  # (bsize, item_cnt, width, width)
                    for j in range(item_cnt):
                        final_masks[:, j] = (max_idx == j).float() * cross_mask[:, j]
                    cross_mask = final_masks
                    cross_mask = F.interpolate(cross_mask, (64, 64))  # mode: nearest

                state_t = state_prev_t

    images = latents_decode(pipeline, latents_t, accelerator.device, prompt_embeds1_combine.dtype).cpu().detach()

    os.makedirs(img_save_dir, exist_ok=True)
    for j, image in enumerate(images):
        pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        pil.save(os.path.join(img_save_dir, f"{(j + global_idx):05}_AsynDM.png"))
