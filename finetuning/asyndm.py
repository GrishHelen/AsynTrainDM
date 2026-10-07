import gc
import json
import os
import os.path
from functools import partial
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from diffusion.asyn_ddim_with_logprob import latents_encode
from finetuning.attention_masks import cross_attention_maps_to_object_mask
from finetuning.eval import val_epoch
from finetuning.metrics import evaluate_epoch_metrics
from finetuning.utils import add_noise, generate_ltg_timesteps_tensor, predict_noise
from model.unet_2d_condition import unet_asyn_forward
from sampling.utils import func_prev_linear, func_prev_binary, item_word_indices_to_token_groups

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

ATTENTION_MASK_LAYER_SIZES = (16, 32)


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


def get_attention_item_file(config):
    return config.finetune.get("item_idx_file", "") or config.item_idx_file


def load_attention_item_maps(config):
    item_idx_file = get_attention_item_file(config)
    if not item_idx_file:
        raise ValueError(
            "finetune_mask_source='attention' requires --finetune_items_file "
            "or --items_file with prompt-to-object token markup"
        )
    with open(item_idx_file, "r", encoding="utf-8") as f:
        item_config = json.load(f)
    if "item_idx" not in item_config:
        raise ValueError(f"{item_idx_file} must contain an 'item_idx' field")
    return item_config["item_idx"], item_config.get("item_k", {})


def get_prompt_item_info(config, prompt, item_idx_by_prompt, item_k_by_prompt):
    if prompt not in item_idx_by_prompt:
        raise KeyError(
            f"Prompt is missing from {get_attention_item_file(config)}: {prompt[:120]!r}"
        )

    item_idx_list = [int(item) for item in item_idx_by_prompt[prompt]]
    item_k_list = item_k_by_prompt.get(prompt)
    if item_k_list is None:
        item_k_list = [config.finetune.item_k] * len(item_idx_list)
    item_k_list = [float(item_k) for item_k in item_k_list]

    if len(item_idx_list) != len(item_k_list):
        raise ValueError(
            f"Different item_idx/item_k lengths for prompt {prompt[:120]!r}: "
            f"{len(item_idx_list)} vs {len(item_k_list)}"
        )
    return item_idx_list, item_k_list


def cross_attention_to_object_mask(config, cross_mask, item_k_list, target_size=64):
    del item_k_list  # Training uses the union of all object regions, independent of schedule priority.
    threshold_type = config.finetune.get("attn_mask_threshold_type", "robust")
    return cross_attention_maps_to_object_mask(
        cross_mask,
        target_size=target_size,
        threshold_type=threshold_type,
        threshold_scale=config.mask_thr,
    )


def build_attention_object_masks(
        config,
        accelerator,
        pipeline,
        latents,
        noise,
        prompt_embeds,
        prompts,
        step,
        item_idx_by_prompt,
        item_k_by_prompt,
):
    masks = []
    num_train_timesteps = pipeline.scheduler.config.num_train_timesteps
    probe_t_value = max(num_train_timesteps - int(step) - 1, 0)

    for batch_idx, prompt in enumerate(prompts):
        item_idx_list, item_k_list = get_prompt_item_info(
            config, prompt, item_idx_by_prompt, item_k_by_prompt
        )
        item_token_groups = item_word_indices_to_token_groups(pipeline.tokenizer, prompt, item_idx_list)
        if len(item_idx_list) == 0:
            masks.append(torch.zeros(64, 64, device=accelerator.device))
            continue

        probe_t = torch.full(
            (1, 64, 64),
            probe_t_value,
            device=accelerator.device,
            dtype=torch.long,
        )
        noisy_latent = add_noise(
            pipeline.scheduler,
            latents[batch_idx:batch_idx + 1],
            noise[batch_idx:batch_idx + 1],
            probe_t,
        )

        # unet_asyn_forward extracts the second half of a CFG-shaped batch.
        latent_input = torch.cat([noisy_latent, noisy_latent], dim=0)
        latent_input = pipeline.scheduler.scale_model_input(latent_input)
        concat_t = torch.cat([probe_t.reshape(1, -1), probe_t.reshape(1, -1)], dim=0)
        prompt_input = torch.cat(
            [prompt_embeds[batch_idx:batch_idx + 1], prompt_embeds[batch_idx:batch_idx + 1]],
            dim=0,
        )

        _, extra_inf = unet_asyn_forward(
            pipeline.unet,
            latent_input,
            concat_t,
            encoder_hidden_states=prompt_input,
            return_dict=False,
            extra_input={
                "used_layer_sizes": ATTENTION_MASK_LAYER_SIZES,
                "item_idx": item_token_groups,
            },
            return_extra_inf=True,
        )
        object_mask = cross_attention_to_object_mask(
            config,
            extra_inf["cross_masks"],
            item_k_list,
            target_size=64,
        )
        masks.append(object_mask.squeeze(0))

    return torch.stack(masks, dim=0)


def train_epoch_asyndm(config, accelerator, pipeline, dataloader, optimizer, epoch):
    autocast = accelerator.autocast
    params_to_optimize = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
    pipeline.unet.train()
    state_stat = [1e7,-1e7,0, 0] # min, max, sum, cnt  
    mask_source = config.finetune.get("mask_source", "dataset")
    if mask_source == "attention" and not config.finetune.get("use_ltg", False):
        item_idx_by_prompt, item_k_by_prompt = load_attention_item_maps(config)
    else:
        item_idx_by_prompt, item_k_by_prompt = None, None

    for i, batch in enumerate(dataloader):
        if i == config.finetune.max_batches:
            break
        with accelerator.accumulate(pipeline.unet):
            with autocast():
                with torch.no_grad():
                    # get clear latents from clear images
                    latents = latents_encode(pipeline, batch["image"].to(accelerator.device))

                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                    noise = torch.randn_like(latents, device=accelerator.device)

                    if config.finetune.get("use_ltg", False):
                        state_t = generate_ltg_timesteps_tensor(config, pipeline, batch_size=latents.shape[0])
                    else:
                        step = np.random.randint(0, pipeline.scheduler.config.num_train_timesteps)
                        if mask_source == "attention":
                            if "prompt" not in batch:
                                raise ValueError("Attention masks require raw prompts in the dataloader batch")
                            prompts = batch["prompt"]
                            if isinstance(prompts, str):
                                prompts = [prompts]
                            cross_mask = build_attention_object_masks(
                                config,
                                accelerator,
                                pipeline,
                                latents,
                                noise,
                                prompt_embeds,
                                prompts,
                                step,
                                item_idx_by_prompt,
                                item_k_by_prompt,
                            )
                        else:
                            if "mask" not in batch:
                                raise ValueError(
                                    "finetune_mask_source='dataset' requires --finetune_use_mask 1 "
                                    "and a dataset with a 'mask' column"
                                )
                            cross_mask = torch.as_tensor(
                                batch['mask'],
                                dtype=torch.float32,
                                device=accelerator.device,
                            )
                        state_t = compute_state_t(config, accelerator, pipeline, cross_mask, step, epoch)
                    state_stat = [min(state_stat[0], torch.min(state_t)), 
                                  max(state_stat[1], torch.max(state_t)),
                                  state_stat[2] + torch.sum(state_t),
                                  state_stat[3] + state_t.numel()]

                    # get noisy_latents from clear latents
                    noisy_latents = add_noise(pipeline.scheduler, latents, noise, state_t)

                # predict noise
                noise_pred = predict_noise(config, pipeline, noisy_latents, state_t, prompt_embeds)

                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(params_to_optimize, config.finetune.max_grad_norm).item()
                optimizer.step()
                optimizer.zero_grad()

    print(f'state_t. min: {round(float(state_stat[0]), 3)}, max: {round(float(state_stat[1]), 3)}, \
          mean: {round(float(state_stat[2]/state_stat[3]), 3)}')
    return loss.item()


def train_asyndm(config, accelerator, pipeline, optimizer, save_dir, train_dataloader):
    best_model_path = None
    models_save_dir = os.path.join(save_dir, "models_state_dict/")
    eval_save_dir = os.path.join(save_dir, "eval_images/")
    os.makedirs(models_save_dir, exist_ok=True)
    os.makedirs(eval_save_dir, exist_ok=True)

    n_epochs = config.finetune.n_epochs
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch + 1}', flush=True)

        train_loss = train_epoch_asyndm(config, accelerator, pipeline, train_dataloader, optimizer, epoch)

        if epoch % config.logging.eval_epoch == 0:
            with torch.no_grad():
                val_epoch(config, accelerator, pipeline, os.path.join(eval_save_dir, f"epoch_{epoch + 1}/"))

                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(models_save_dir, f'model_{epoch + 1}.pth')
                torch.save(pipeline.unet.state_dict(), best_model_path)

        evaluate_epoch_metrics(config, accelerator, pipeline, save_dir, epoch)

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
