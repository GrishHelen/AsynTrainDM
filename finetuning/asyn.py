import os.path
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from diffusion.asyn_ddim_with_logprob import latents_encode
from finetuning.utils import generate_timesteps
from model.unet_2d_condition import unet_asyn_forward
from utils.sampling import prepare_encoded_prompts, encode_prompts_list

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


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


def train_epoch_asyn(config, accelerator, pipeline, dataloader, optimizer, sample_neg_prompt_embeds):
    autocast = accelerator.autocast
    losses = []
    pipeline.unet.train()

    for batch in dataloader:
        with autocast():
            with torch.no_grad():
                # get clear latents from clear images
                latents = latents_encode(pipeline, batch["images"])

                # generate prompts
                prompt_embeds1_combine = prepare_encoded_prompts(config, accelerator, pipeline, batch["prompts"],
                                                                 sample_neg_prompt_embeds)

            timesteps = generate_timesteps(config, pipeline)

            # get noisy_latents from clear latents
            noise = torch.randn_like(latents)
            noisy_latents = pipeline.scheduler.add_noise(
                latents,
                noise,
                timesteps
            )
            noise_real = latents - noisy_latents

            # predict noise
            noise_pred = predict_noise(config, pipeline, noisy_latents, timesteps, prompt_embeds1_combine)

            loss = F.mse_loss(noise_pred, noise_real)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return losses


def val_epoch_asyn(config, accelerator, pipeline, dataloader, optimizer, sample_neg_prompt_embeds):
    autocast = accelerator.autocast
    losses = []
    pipeline.unet.eval()

    for batch in dataloader:
        with autocast():
            with torch.no_grad():
                # get clear latents from clear images
                latents = pipeline.vae.encode(batch["images"]).sample
                latents = latents * pipeline.vae.config.scaling_factor

                # generate prompts
                prompt_embeds1_combine = prepare_encoded_prompts(config, accelerator, pipeline, batch["prompts"],
                                                                 sample_neg_prompt_embeds)

                timesteps = generate_timesteps(config, pipeline)

                # get noisy_latents from clear latents
                noise = torch.randn_like(latents)
                noisy_latents = pipeline.scheduler.add_noise(
                    latents,
                    noise,
                    timesteps
                )
                noise_real = latents - noisy_latents

                # predict noise
                noise_pred = predict_noise(config, pipeline, noisy_latents, timesteps, prompt_embeds1_combine)

                loss = F.mse_loss(noise_pred, noise_real)
                losses.append(loss.item())

    return losses


def train_asyn(config, accelerator, pipeline, train_dataloader, val_dataloader=None,
               optimizer=None, sample_neg_prompt_embeds=None, models_save_dir=None):
    pipeline.unet.train()
    best_val_loss = 1e6

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            pipeline.unet.parameters()
        )

    if sample_neg_prompt_embeds is None:
        sample_neg_prompt_embeds = encode_prompts_list(pipeline, accelerator.device, [""])

    n_epochs = config.finetune.n_epochs
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch}')

        train_losses = train_epoch_asyn(config, accelerator, pipeline, train_dataloader, optimizer,
                                        sample_neg_prompt_embeds)
        avg_train_loss = np.mean(train_losses)
        print(f'train loss: {round(avg_train_loss, 3)}')

        if val_dataloader is not None:
            with torch.no_grad():
                val_losses = train_epoch_asyn(config, accelerator, pipeline, val_dataloader, optimizer,
                                              sample_neg_prompt_embeds)
                avg_val_loss = np.mean(val_losses)
                print(f'val loss: {round(avg_val_loss, 3)}')

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_path = os.path.join(models_save_dir, 'models_state_dict', f'model_{epoch}.pth')
                    torch.save(pipeline.unet.state_dict(), model_path)

    model_path = os.path.join(models_save_dir, 'models_state_dict', f'model_{n_epochs}_res.pth')
    torch.save(pipeline.unet.state_dict(), model_path)
