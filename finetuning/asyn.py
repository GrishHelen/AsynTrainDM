import os.path
from functools import partial

import torch
import torch.nn.functional as F
import tqdm
from torch.nn.utils import clip_grad_norm_

from diffusion.asyn_ddim_with_logprob import latents_encode
from finetuning.utils import generate_timesteps_tensor, add_noise, predict_noise, array_to_file
from utils.sampling import encode_prompts_list

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def train_epoch_asyn(config, accelerator, pipeline, dataloader, optimizer, sample_neg_prompt_embeds, models_save_dir):
    autocast = accelerator.autocast
    losses = []
    grad_norms = []
    pipeline.unet.train()

    for i, batch in enumerate(dataloader):
        with autocast():
            with torch.no_grad():
                # get clear latents from clear images
                latents = latents_encode(pipeline, batch["image"].to(accelerator.device))

                # generate prompts
                prompt_embeds1_combine = torch.cat([sample_neg_prompt_embeds[:latents.shape[0]],
                                                    batch["prompt_embeds"].to(accelerator.device)], dim=0)

                ts_tensor = generate_timesteps_tensor(pipeline, batch_size=latents.shape[0])

                # get noisy_latents from clear latents
                noisy_latents = add_noise(pipeline, latents, ts_tensor)
                noise_real = latents - noisy_latents

            # predict noise
            noise_pred = predict_noise(config, pipeline, noisy_latents, ts_tensor, prompt_embeds1_combine)

            loss = F.mse_loss(noise_pred, noise_real)
            losses.append(round(loss.item(), 4))
            loss.backward()
            total_norm = clip_grad_norm_(pipeline.unet.parameters(), max_norm=config.finetune.max_grad_norm).item()
            grad_norms.append(total_norm)
            if (i + 1) % config.finetune.grad_accumulation_steps == 0 or i == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            if losses[-1] is None:
                return None

            if i % config.logging.batch == 0:
                array_to_file(models_save_dir, 'train_loss_history.txt', losses)
                losses = []
                array_to_file(models_save_dir, 'grad_norms_history.txt', grad_norms)
                grad_norms = []

    array_to_file(models_save_dir, 'train_loss_history.txt', losses)
    array_to_file(models_save_dir, 'grad_norms_history.txt', grad_norms)
    return loss.item()


def val_epoch_asyn(config, accelerator, pipeline, dataloader, sample_neg_prompt_embeds, models_save_dir):
    autocast = accelerator.autocast
    losses = []
    pipeline.unet.eval()

    for i, batch in enumerate(dataloader):
        with autocast():
            with torch.no_grad():
                # get clear latents from clear images
                latents = latents_encode(pipeline, batch["image"].to(accelerator.device))

                # generate prompts
                prompt_embeds1_combine = torch.cat([sample_neg_prompt_embeds[:latents.shape[0]],
                                                    batch["prompt_embeds"].to(accelerator.device)], dim=0)
                ts_tensor = generate_timesteps_tensor(pipeline, batch_size=latents.shape[0])

                # get noisy_latents from clear latents
                noisy_latents = add_noise(pipeline, latents, ts_tensor)
                noise_real = latents - noisy_latents

                # predict noise
                noise_pred = predict_noise(config, pipeline, noisy_latents, ts_tensor, prompt_embeds1_combine)

                loss = F.mse_loss(noise_pred, noise_real)
                losses.append(round(loss.item(), 4))
                if i % config.logging.batch == 0:
                    array_to_file(models_save_dir, 'val_loss_history.txt', losses)
                    losses = []
    array_to_file(models_save_dir, 'val_loss_history.txt', losses)
    return loss.item()


def train_asyn(config, accelerator, pipeline, optimizer, models_save_dir, train_dataloader, val_dataloader=None,
               sample_neg_prompt_embeds=None):
    best_val_loss = (1e6, '')
    os.makedirs(os.path.join(models_save_dir, "models_state_dict/"), exist_ok=True)
    loss_history = []

    if sample_neg_prompt_embeds is None:
        neg_prompt_embed = encode_prompts_list(pipeline, accelerator.device, [""])
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.finetune.batch_size, 1, 1)

    n_epochs = config.finetune.n_epochs
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch + 1}', flush=True)

        train_loss = train_epoch_asyn(config, accelerator, pipeline, train_dataloader, optimizer,
                                      sample_neg_prompt_embeds, models_save_dir)
        loss_history.append([train_loss])
        print(f'train loss: {train_loss}', flush=True)

        if val_dataloader is not None:
            with torch.no_grad():
                val_loss = val_epoch_asyn(config, accelerator, pipeline, val_dataloader,
                                          sample_neg_prompt_embeds, models_save_dir)
                loss_history[-1].append(val_loss)
                print(f'val loss: {val_loss}', flush=True)
                if val_loss is None:
                    raise AssertionError(f'val loss is {val_loss}')

                if val_loss < best_val_loss[0]:
                    if len(best_val_loss[1]):
                        os.remove(best_val_loss[1])
                    model_path = os.path.join(models_save_dir, 'models_state_dict', f'model_{epoch + 1}.pth')
                    best_val_loss = (val_loss, model_path)
                    torch.save(pipeline.unet.state_dict(), model_path)
        if epoch % config.logging.epoch == 0:
            array_to_file(models_save_dir, 'epoch_loss_history.txt', loss_history)
            loss_history = []
        if train_loss is None:
            return

    array_to_file(models_save_dir, 'epoch_loss_history.txt', loss_history)

    model_path = os.path.join(models_save_dir, 'models_state_dict', f'model_{n_epochs}_res.pth')
    torch.save(pipeline.unet.state_dict(), model_path)
