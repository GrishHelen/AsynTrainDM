import gc
import os.path
from functools import partial

import torch
import torch.nn.functional as F
import tqdm

from diffusion.asyn_ddim_with_logprob import latents_encode
from finetuning.eval import val_epoch
from finetuning.metrics import evaluate_epoch_metrics
from finetuning.utils import add_noise, generate_ltg_timesteps_tensor, generate_timesteps_tensor, predict_noise

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def train_epoch_asyn(config, accelerator, pipeline, dataloader, optimizer):
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

                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device)

                    if config.finetune.get("use_ltg", False):
                        ts_tensor = generate_ltg_timesteps_tensor(config, pipeline, batch_size=latents.shape[0])
                    else:
                        ts_tensor = generate_timesteps_tensor(pipeline, batch_size=latents.shape[0],
                                                              type=config.finetune.ts_type)
                    noise = torch.randn_like(latents, device=accelerator.device)

                    # get noisy_latents from clear latents
                    noisy_latents = add_noise(pipeline.scheduler, latents, noise, ts_tensor)

                # predict noise
                noise_pred = predict_noise(config, pipeline, noisy_latents, ts_tensor, prompt_embeds)

                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(params_to_optimize, config.finetune.max_grad_norm).item()
                optimizer.step()
                optimizer.zero_grad()

    return loss.item()


def train_asyn(config, accelerator, pipeline, optimizer, save_dir, train_dataloader):
    best_model_path = None
    models_save_dir = os.path.join(save_dir, "models_state_dict/")
    eval_save_dir = os.path.join(save_dir, "eval_images/")
    os.makedirs(models_save_dir, exist_ok=True)
    os.makedirs(eval_save_dir, exist_ok=True)

    n_epochs = config.finetune.n_epochs
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch + 1}', flush=True)

        train_loss = train_epoch_asyn(config, accelerator, pipeline, train_dataloader, optimizer)

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
