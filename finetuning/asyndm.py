import gc
import os
import os.path
from functools import partial

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


def compute_state_t(config, accelerator, pipeline, cross_mask, step):
    initial_t = pipeline.scheduler.config.num_train_timesteps + pipeline.scheduler.config.steps_offset
    initial_t = torch.tensor(initial_t, device=accelerator.device, dtype=torch.float32)
    state_t = initial_t[None, None, None].expand(cross_mask.shape[0], 64, 64)

    bg_mask = 1 - (cross_mask > 0.5).float()
    state_prev_t_linear = func_prev_linear(pipeline, state_t, config.sample.num_steps)
    prev_binary_val = func_prev_binary(config, pipeline, state_t, config.sample.num_steps, k=0.7)
    state_prev_t_binary = cross_mask * prev_binary_val
    state_t = (bg_mask * state_prev_t_linear + state_prev_t_binary).round().long()

    for i in range(step):
        state_prev_t_linear = func_prev_linear(pipeline, state_t, config.sample.num_steps - i - 1)
        state_prev_t_binary = cross_mask * func_prev_binary(config, pipeline,
                                                            state_t, config.sample.num_steps - i - 1,
                                                            k=0.7)
        state_prev_t = (bg_mask * state_prev_t_linear + state_prev_t_binary)
        state_t = state_prev_t.round().long()

    return state_t


def train_epoch_asyndm(config, accelerator, pipeline, dataloader, optimizer, sample_neg_prompt_embeds, epoch):
    autocast = accelerator.autocast
    params_to_optimize = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
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

                    cross_mask = torch.tensor(batch['mask'], dtype=torch.float32)
                    step = np.random.randint(0, config.sample.num_steps)
                    state_t = compute_state_t(config, accelerator, pipeline, cross_mask, step)
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
