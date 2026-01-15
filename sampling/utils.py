import torch
import numpy as np
import json


def get_item_idx_list(config, prompt_idx):
    item_idx_list = [config.item_idx] if isinstance(config.prompt, str) else config.item_idx[prompt_idx]
    if len(config.item_idx_file) != 0:
        with open(config.item_idx_file, 'r') as f:
            temp_list = json.load(f)
            item_idx_list = temp_list["item_idx"][prompt_idx]
    return item_idx_list


def get_item_k_list(config, prompt_idx):
    item_k_list = config.item_k if isinstance(config.prompt, str) else config.item_k[prompt_idx]
    if len(config.item_idx_file) != 0:
        with open(config.item_idx_file, 'r') as f:
            temp_list = json.load(f)
            item_k_list = temp_list["item_k"][prompt_idx]
    return item_k_list


def func_prev_linear(pipeline, state_t, rest_step, target_value=None):
    if target_value is None:
        target_value = pipeline.scheduler.config.steps_offset

    if rest_step == 0:
        state_prev_t = torch.zeros_like(state_t)
    else:
        target_value_tensor = torch.tensor(target_value, dtype=state_t.dtype, device=state_t.device)
        state_prev_t = (state_t * (1 - 1 / rest_step) + target_value_tensor / rest_step)
    return state_prev_t


def func_prev_binary(
        config, pipeline,
        state_t, rest_step, k=0.5,
        target_value=None,
        curve_type=None,
        x_scaling=None,
        y_scaling=None
):
    if target_value is None:
        target_value = pipeline.scheduler.config.steps_offset
    if curve_type is None:
        curve_type = config.curve_type
    if x_scaling is None:
        x_scaling = config.sample.num_steps
    if y_scaling is None:
        y_scaling = pipeline.scheduler.config.num_train_timesteps

    if rest_step == 0:
        state_prev_t = torch.zeros_like(state_t)
    else:
        if curve_type == "bin":
            decay_k = y_scaling / (x_scaling * x_scaling)
            k = -k * decay_k
            target_value_tensor = torch.tensor(target_value, dtype=state_t.dtype, device=state_t.device)
            t0 = (k * rest_step ** 2 - (target_value_tensor - state_t)) / (2 * k * rest_step)
            y0 = state_t - k * t0 ** 2
            state_prev_t = (k * (1 - t0) ** 2 + y0)
        elif curve_type == "lin":
            k1 = -(1 - k) * y_scaling / x_scaling
            k2 = -(1 + k) * y_scaling / x_scaling
            c1 = state_t
            c2 = target_value - k2 * rest_step
            state_prev_t = (k1 + c1).clamp(max=k2 + c2)
        elif curve_type == "exp":
            # y = a*e^(lamb*x)+bx+c
            lamb = 1 / x_scaling
            a = -k * y_scaling / (np.e - 1)
            b = -(1 - k) * y_scaling / x_scaling
            # state_prev_t = k*(np.e**lamb)+a+b
            exp_neg_lamb_x0 = (state_t - target_value + b * rest_step) / (
                    a * (1 - np.e ** (lamb * rest_step)))  # e^(-lamb*x_0)
            state_prev_t = a * (np.e ** lamb - 1) * exp_neg_lamb_x0 + b + state_t
        else:
            raise ValueError(f"wrong curve type: {curve_type}")
    return state_prev_t
