import gc
import os
import os.path
from functools import partial
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from diffusion.asyn_ddim_with_logprob import latents_encode
from finetuning.eval import val_epoch
from finetuning.utils import add_noise, predict_noise
from sampling.utils import func_prev_linear, func_prev_binary
from utils.sampling import encode_prompts_list

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

class FinetuneWarmupType(Enum):
    POLYNOM = 'polynom'
    MIXTURE = 'mixture'

def compute_state_t(config, accelerator, pipeline, cross_mask, step, epoch):
    def mix_schedules(linear, concave, bg_mask):
        if epoch >= config.finetune.schedule_warmup.n_epochs:
            state_t = (bg_mask * linear + (1 - bg_mask) * concave)
            return state_t
        
        if config.finetune.schedule_warmup.type == FinetuneWarmupType.POLYNOM:
            p = 1 + (epoch / config.finetune.schedule_warmup.n_epochs)
            # TODO
            raise NotImplementedError(f'Schedule_warmup type {config.finetune.schedule_warmup.type.value} not implemented')
        elif config.finetune.schedule_warmup.type == FinetuneWarmupType.MIXTURE:
            mix_coeff = epoch / config.finetune.schedule_warmup.n_epochs
            concave_mixed = (1 - mix_coeff) * linear + mix_coeff * concave
            state_t = (bg_mask * linear + (1 - bg_mask) * concave_mixed)
            return state_t
        else:
            raise NotImplementedError(f'Schedule_warmup type {config.finetune.schedule_warmup.type.value} not implemented')
    
    initial_t = pipeline.scheduler.config.num_train_timesteps + pipeline.scheduler.config.steps_offset
    initial_t = torch.tensor(initial_t, device=accelerator.device, dtype=torch.float32)
    state_t = initial_t[None, None, None].expand(cross_mask.shape[0], 64, 64)

    bg_mask = 1 - (cross_mask > 0.5).float()
    state_prev_t_linear = func_prev_linear(pipeline, state_t, pipeline.scheduler.config.num_train_timesteps)
    state_prev_t_binary = func_prev_binary(config, pipeline, state_t, pipeline.scheduler.config.num_train_timesteps, 
                                           k=config.finetune.item_k,
                                           x_scaling=pipeline.scheduler.config.num_train_timesteps,
                                           y_scaling=pipeline.scheduler.config.num_train_timesteps)
    state_t = mix_schedules(state_prev_t_linear, state_prev_t_binary, bg_mask).round().long()


    for i in range(step):
        state_prev_t_linear = func_prev_linear(pipeline, state_t, pipeline.scheduler.config.num_train_timesteps - i - 1)
        state_prev_t_binary = func_prev_binary(config, pipeline,
                                                state_t, pipeline.scheduler.config.num_train_timesteps - i - 1,
                                                k=config.finetune.item_k,
                                                x_scaling=pipeline.scheduler.config.num_train_timesteps,
                                                y_scaling=pipeline.scheduler.config.num_train_timesteps)
        state_t = mix_schedules(state_prev_t_linear, state_prev_t_binary, bg_mask).round().long()

    state_t = torch.clamp(state_t, min=0, max=pipeline.scheduler.config.num_train_timesteps - 1)
    return state_t


def train_epoch_asyndm(config, accelerator, pipeline, dataloader, optimizer, sample_neg_prompt_embeds, epoch):
    autocast = accelerator.autocast
    params_to_optimize = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
    pipeline.unet.train()
    state_stat = [1e7,-1e7,0, 0] # min, max, sum, cnt  

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

                    cross_mask = torch.tensor(batch['mask'], dtype=torch.float32)
                    step = np.random.randint(0, pipeline.scheduler.config.num_train_timesteps)
                    state_t = compute_state_t(config, accelerator, pipeline, cross_mask, step, epoch)
                    state_stat = [min(state_stat[0], torch.min(state_t)), 
                                  max(state_stat[1], torch.max(state_t)),
                                  state_stat[2] + torch.sum(state_t),
                                  state_stat[3] + state_t.shape[0]]
                    noise = torch.randn_like(latents, device=accelerator.device)

                    # get noisy_latents from clear latents
                    noisy_latents = add_noise(pipeline.scheduler, latents, noise, state_t)

                # predict noise
                noise_pred = predict_noise(config, pipeline, noisy_latents, state_t, prompt_embeds1_combine)

                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(params_to_optimize, config.finetune.max_grad_norm).item()
                optimizer.step()
                optimizer.zero_grad()

    print(f'state_t. min: {round(float(state_stat[0]), 3)}, max: {round(float(state_stat[1]), 3)}, \
          mean: {round(float(state_stat[2]/state_stat[3]), 3)}')
    return loss.item()


def train_asyndm(config, accelerator, pipeline, optimizer, save_dir, train_dataloader,
                 sample_neg_prompt_embeds=None):
    best_model_path = None
    models_save_dir = os.path.join(save_dir, "models_state_dict/")
    eval_save_dir = os.path.join(save_dir, "eval_images/")
    os.makedirs(models_save_dir, exist_ok=True)
    os.makedirs(eval_save_dir, exist_ok=True)

    if sample_neg_prompt_embeds is None:
        neg_prompt_embed = encode_prompts_list(pipeline, accelerator.device, [""])
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.finetune.batch_size, 1, 1)

    n_epochs = config.finetune.n_epochs
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch + 1}', flush=True)

        train_loss = train_epoch_asyndm(config, accelerator, pipeline, train_dataloader, optimizer,
                                        sample_neg_prompt_embeds, epoch)

        if epoch % config.logging.eval_epoch == 0:
            with torch.no_grad():
                val_epoch(config, accelerator, pipeline, os.path.join(eval_save_dir, f"epoch_{epoch + 1}/"))

                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(models_save_dir, f'model_{epoch + 1}.pth')
                torch.save(pipeline.unet.state_dict(), best_model_path)

        if train_loss is None:
            return

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accelerator.free_memory()

        print(f'\nCompleted epoch {epoch + 1}', flush=True)

    if best_model_path is not None:
        os.remove(best_model_path)
    best_model_path = os.path.join(models_save_dir, f'model_{n_epochs}.pth')
    torch.save(pipeline.unet.state_dict(), best_model_path)
