import gc
import os.path
from functools import partial

import torch
import torch.nn.functional as F
import tqdm
from diffusers import DDIMScheduler

from diffusion.asyn_ddim_with_logprob import latents_encode
from finetuning.utils import generate_timesteps_tensor, add_noise, predict_noise, array_to_file
from sampling.all import sample_all
from utils.sampling import encode_prompts_list

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def train_epoch_asyn(config, accelerator, pipeline, dataloader, optimizer, sample_neg_prompt_embeds, losses_save_dir):
    autocast = accelerator.autocast
    params_to_optimize = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
    losses = []
    grad_norms = []
    pipeline.unet.train()

    for i, batch in enumerate(dataloader):
        if i == config.finetune.max_batches:
            break
        with accelerator.accumulate(pipeline.unet):
            with autocast():
                with torch.no_grad():
                    # get clear latents from clear images
                    latents = latents_encode(pipeline, batch["image"].to(accelerator.device))

                    # generate prompts
                    prompt_embeds1_combine = torch.cat([sample_neg_prompt_embeds[:latents.shape[0]],
                                                        batch["prompt_embeds"].to(accelerator.device)], dim=0)

                    ts_tensor = generate_timesteps_tensor(pipeline, batch_size=latents.shape[0],
                                                          type=config.finetune.ts_type)
                    noise = torch.randn_like(latents, device=accelerator.device)

                    # get noisy_latents from clear latents
                    noisy_latents = add_noise(pipeline.scheduler, latents, noise, ts_tensor)

                # predict noise
                noise_pred = predict_noise(config, pipeline, noisy_latents, ts_tensor, prompt_embeds1_combine)

                loss = F.mse_loss(noise_pred, noise)
                losses.append(round(loss.item(), 4))

                if torch.isnan(loss).item():
                    array_to_file(losses_save_dir, 'train_loss_history.txt', losses)
                    array_to_file(losses_save_dir, 'grad_norms_history.txt', grad_norms)
                    return None

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(params_to_optimize, config.finetune.max_grad_norm).item()
                    grad_norms.append(total_norm)
                optimizer.step()
                optimizer.zero_grad()

        if i % config.logging.batch == 0:
            array_to_file(losses_save_dir, 'train_loss_history.txt', losses)
            losses = []
            array_to_file(losses_save_dir, 'grad_norms_history.txt', grad_norms)
            grad_norms = []

    array_to_file(losses_save_dir, 'train_loss_history.txt', losses)
    array_to_file(losses_save_dir, 'grad_norms_history.txt', grad_norms)
    return loss.item()


# def val_epoch_asyn(config, accelerator, pipeline, dataloader, sample_neg_prompt_embeds, models_save_dir):
#     autocast = accelerator.autocast
#     losses = []
#     pipeline.unet.eval()
#
#     for i, batch in enumerate(dataloader):
#         with autocast():
#             with torch.no_grad():
#                 # get clear latents from clear images
#                 latents = latents_encode(pipeline, batch["image"].to(accelerator.device))
#
#                 # generate prompts
#                 prompt_embeds1_combine = torch.cat([sample_neg_prompt_embeds[:latents.shape[0]],
#                                                     batch["prompt_embeds"].to(accelerator.device)], dim=0)
#                 ts_tensor = generate_timesteps_tensor(pipeline, batch_size=latents.shape[0],
#                                                       type=config.finetune.ts_type)
#                 noise = torch.randn_like(latents, device=accelerator.device)
#
#                 # get noisy_latents from clear latents
#                 noisy_latents = add_noise(pipeline.scheduler, latents, noise, ts_tensor)
#
#                 # predict noise
#                 noise_pred = predict_noise(config, pipeline, noisy_latents, ts_tensor, prompt_embeds1_combine)
#
#                 loss = F.mse_loss(noise_pred, noise)
#                 losses.append(round(loss.item(), 4))
#                 if i % config.logging.batch == 0:
#                     array_to_file(models_save_dir, 'val_loss_history.txt', losses)
#                     losses = []
#     array_to_file(models_save_dir, 'val_loss_history.txt', losses)
#     return loss.item()

def val_epoch_asyn(config, accelerator, pipeline, images_save_dir):
    autocast = accelerator.autocast
    pipeline.unet.eval()
    with autocast():
        with torch.no_grad():
            old_scheduler = pipeline.scheduler
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

            sample_all(config, accelerator, pipeline, img_save_dir=images_save_dir)

            pipeline.scheduler = old_scheduler


def train_asyn(config, accelerator, pipeline, optimizer, save_dir, train_dataloader,
               sample_neg_prompt_embeds=None):
    best_model_path = None
    models_save_dir = os.path.join(save_dir, "models_state_dict/")
    eval_save_dir = os.path.join(save_dir, "eval_images/")
    os.makedirs(models_save_dir, exist_ok=True)
    os.makedirs(eval_save_dir, exist_ok=True)
    loss_history = []

    if sample_neg_prompt_embeds is None:
        neg_prompt_embed = encode_prompts_list(pipeline, accelerator.device, [""])
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.finetune.batch_size, 1, 1)

    n_epochs = config.finetune.n_epochs
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch + 1}', flush=True)

        train_loss = train_epoch_asyn(config, accelerator, pipeline, train_dataloader, optimizer,
                                      sample_neg_prompt_embeds, save_dir)
        loss_history.append([train_loss])
        print(f'train loss: {train_loss}', flush=True)

        if epoch % config.logging.eval_epoch == 0:
            with torch.no_grad():
                val_epoch_asyn(config, accelerator, pipeline, os.path.join(eval_save_dir, f"epoch_{epoch + 1}/"))

                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(models_save_dir, f'model_{epoch + 1}.pth')
                torch.save(pipeline.unet.state_dict(), best_model_path)
        if epoch % config.logging.epoch == 0:
            array_to_file(save_dir, 'epoch_loss_history.txt', loss_history)
            loss_history = []
        if train_loss is None:
            return

        gc.collect()
        torch.cuda.empty_cache()
        accelerator.free_memory()

    array_to_file(save_dir, 'epoch_loss_history.txt', loss_history)

    if best_model_path is not None:
        os.remove(best_model_path)
    best_model_path = os.path.join(models_save_dir, f'model_{n_epochs}.pth')
    torch.save(pipeline.unet.state_dict(), best_model_path)
