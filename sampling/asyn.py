import os
import re
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageOps

from diffusion.asyn_ddim_with_logprob import asyn_ddim_step_with_logprob, latents_decode
from model.unet_2d_condition import unet_asyn_forward
from .utils import (
    get_item_idx_list,
    get_item_k_list,
    func_prev_linear,
    func_prev_binary,
    item_word_indices_to_token_groups,
    item_word_labels,
)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


RESAMPLE_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
RESAMPLE_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


def tensor_image_to_pil(image_tensor):
    image = image_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    return Image.fromarray((image * 255).astype(np.uint8))


def safe_filename(text, max_len=140):
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._ ")
    if not text:
        text = "prompt"
    return text[:max_len]


def get_prompt_text(config, prompt_idx, prompt=None):
    if prompt is not None:
        return prompt
    if len(config.prompt_file) != 0:
        import json
        with open(config.prompt_file, "r", encoding="utf-8") as f:
            prompt_list = json.load(f)
        return prompt_list[prompt_idx]
    return config.prompt if isinstance(config.prompt, str) else config.prompt[prompt_idx]


def attention_save_steps(config):
    every = int(config.sample.get("attn_grid_every", 5))
    if every <= 0:
        return {config.sample.num_steps - 1}
    steps = set(range(0, config.sample.num_steps, every))
    steps.add(config.sample.num_steps - 1)
    return steps


def cross_attention_maps(config, cross_mask, item_k_list, target_size=64):
    if cross_mask is None:
        return None, None

    cross_mask = cross_mask.detach().float()
    bsize, width_height, item_cnt = cross_mask.shape
    width = int(width_height ** 0.5)
    if width * width != width_height:
        raise ValueError(f"Cross-attention map size must be square, got HW={width_height}")

    raw_maps = cross_mask.permute(0, 2, 1).reshape(bsize, item_cnt, width, width)
    mask_mean = config.mask_thr * cross_mask.mean(dim=1, keepdim=True)
    binary_maps = (cross_mask >= mask_mean).float()
    binary_maps = binary_maps.permute(0, 2, 1).reshape(bsize, item_cnt, width, width)

    priority = torch.tensor(item_k_list, dtype=torch.float32, device=cross_mask.device).view(1, item_cnt, 1, 1)
    priority_masks = binary_maps * priority
    _, max_idx = priority_masks.max(dim=1)

    final_masks = torch.zeros_like(binary_maps)
    for item_idx in range(item_cnt):
        final_masks[:, item_idx] = (max_idx == item_idx).float() * binary_maps[:, item_idx]

    if raw_maps.shape[-2:] != (target_size, target_size):
        raw_maps = F.interpolate(raw_maps, (target_size, target_size), mode="bilinear", align_corners=False)
    if final_masks.shape[-2:] != (target_size, target_size):
        final_masks = F.interpolate(final_masks, (target_size, target_size), mode="nearest")
    return raw_maps, final_masks


def map_to_grid_image(map_tensor, cell_size, mode, value_range=None):
    array = map_tensor.detach().cpu().float().numpy()
    if mode == "raw":
        if value_range is None:
            vmin = float(np.min(array))
            vmax = float(np.max(array))
        else:
            vmin, vmax = value_range
        if vmax > vmin:
            array = (array - vmin) / (vmax - vmin)
        else:
            array = np.zeros_like(array)
        image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L")
        image = image.resize((cell_size, cell_size), RESAMPLE_BILINEAR)
        return ImageOps.colorize(image, black="#111827", white="#facc15")

    image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L")
    image = image.resize((cell_size, cell_size), RESAMPLE_NEAREST)
    return ImageOps.colorize(image, black="#111827", white="#34d399")


def draw_text(draw, xy, text, max_width, font, fill="#111827", line_height=13):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines[:4]:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height


def save_attention_grid(config, img_save_dir, global_idx, prompt, item_labels, records):
    if not records:
        return

    cell_size = int(config.sample.get("attn_grid_cell_size", 192))
    label_width = 180
    title_height = 54
    header_height = 32
    row_gap = 8
    font = ImageFont.load_default()

    rows = [("image", "image", None)]
    for item_idx, label in enumerate(item_labels):
        rows.append((f"raw {label}", "raw", item_idx))
    for item_idx, label in enumerate(item_labels):
        rows.append((f"mask {label}", "mask", item_idx))
    rows.append(("union mask", "union", None))

    cols = len(records)
    grid_width = label_width + cols * cell_size
    grid_height = title_height + header_height + len(rows) * (cell_size + row_gap)
    grid = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(grid)

    prompt_title = prompt if len(prompt) <= 220 else f"{prompt[:217]}..."
    draw_text(draw, (10, 8), prompt_title, grid_width - 20, font, fill="#111827", line_height=13)

    y0 = title_height
    for col_idx, record in enumerate(records):
        x = label_width + col_idx * cell_size
        step_text = f"step {record['step']}"
        bbox = draw.textbbox((0, 0), step_text, font=font)
        draw.text((x + (cell_size - (bbox[2] - bbox[0])) // 2, y0 + 10), step_text, font=font, fill="#111827")

    y = title_height + header_height
    for label, row_type, item_idx in rows:
        draw_text(draw, (10, y + 8), label, label_width - 20, font, fill="#111827", line_height=13)

        value_range = None
        if row_type == "raw":
            values = [record["raw_maps"][item_idx] for record in records]
            value_range = (min(float(torch.min(value)) for value in values),
                           max(float(torch.max(value)) for value in values))

        for col_idx, record in enumerate(records):
            x = label_width + col_idx * cell_size
            if row_type == "image":
                cell = record["image"].resize((cell_size, cell_size), RESAMPLE_BILINEAR)
            elif row_type == "raw":
                cell = map_to_grid_image(record["raw_maps"][item_idx], cell_size, "raw", value_range=value_range)
            elif row_type == "mask":
                cell = map_to_grid_image(record["final_masks"][item_idx], cell_size, "mask")
            else:
                union = (record["final_masks"] > 0.5).any(dim=0).float()
                cell = map_to_grid_image(union, cell_size, "mask")

            grid.paste(cell, (x, y))
            draw.rectangle((x, y, x + cell_size - 1, y + cell_size - 1), outline="#d1d5db")

        y += cell_size + row_gap

    attn_grid_dir = os.path.join(img_save_dir, config.sample.get("attn_grid_dir", "attn_grids"))
    os.makedirs(attn_grid_dir, exist_ok=True)
    file_name = f"{global_idx:05}_{safe_filename(prompt)}.png"
    grid.save(os.path.join(attn_grid_dir, file_name))


def generate_asyn(config, accelerator, pipeline, idx, prompt_embeds1_combine, cross_mask=None, img_save_dir=None,
                  prompt=None):
    global_idx = idx * config.sample.batch_size
    autocast = accelerator.autocast
    prompt_idx = idx // config.sample.num_batches_per_epoch
    if img_save_dir is None:
        img_save_dir = os.path.join(accelerator.project_configuration.project_dir, "images/")
    prompt_text = get_prompt_text(config, prompt_idx, prompt=prompt)
    item_idx_list = get_item_idx_list(config, prompt_idx)
    item_k_list = get_item_k_list(config, prompt_idx)
    item_token_groups = item_word_indices_to_token_groups(pipeline.tokenizer, prompt_text, item_idx_list)
    save_attn_grids = bool(config.sample.get("save_attn_grids", False))
    attn_steps = attention_save_steps(config) if save_attn_grids else set()
    item_labels = item_word_labels(prompt_text, item_idx_list, item_token_groups) if save_attn_grids else []
    attn_records = [[] for _ in range(config.sample.batch_size)] if save_attn_grids else None

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
    max_timestep = pipeline.scheduler.config.num_train_timesteps - 1
    state_t = (bg_mask * state_prev_t_linear + state_prev_t_binary).clamp(0, max_timestep)
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
                state_prev_t = (bg_mask * state_prev_t_linear + state_prev_t_binary).clamp(0, max_timestep)

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
                                                              'item_idx': item_token_groups
                                                          },
                                                          return_extra_inf=True,
                                                          )
                noise_pred = noise_pred[0]
                raw_maps, final_masks = cross_attention_maps(config, extra_inf['cross_mask'], item_k_list)
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

                if save_attn_grids and i in attn_steps and raw_maps is not None:
                    decoded_step_images = latents_decode(
                        pipeline,
                        latents_t,
                        accelerator.device,
                        prompt_embeds1_combine.dtype,
                    ).cpu().detach()
                    for j in range(config.sample.batch_size):
                        attn_records[j].append({
                            "step": i,
                            "image": tensor_image_to_pil(decoded_step_images[j]),
                            "raw_maps": raw_maps[j].cpu(),
                            "final_masks": final_masks[j].cpu(),
                        })

                if not config.static_mask:
                    cross_mask = final_masks

                state_t = state_prev_t

    images = latents_decode(pipeline, latents_t, accelerator.device, prompt_embeds1_combine.dtype).cpu().detach()

    os.makedirs(img_save_dir, exist_ok=True)
    for j, image in enumerate(images):
        pil = tensor_image_to_pil(image)
        pil.save(os.path.join(img_save_dir, f"{(j + global_idx):05}_AsynDM.png"))
        if save_attn_grids:
            save_attention_grid(
                config,
                img_save_dir,
                j + global_idx,
                prompt_text,
                item_labels,
                attn_records[j],
            )
