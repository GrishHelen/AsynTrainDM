import torch


def prepare_encoded_prompts(config, accelerator, pipeline, prompt, sample_neg_prompt_embeds=None):
    if sample_neg_prompt_embeds is None:
        # generate negative prompt embeddings
        neg_prompt_embed = pipeline.text_encoder(
            pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
        )[0]
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)

    prompts1 = [
        prompt
        for _ in range(config.sample.batch_size)
    ]

    # encode prompts
    prompt_ids1 = pipeline.tokenizer(
        prompts1,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(accelerator.device)
    prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]
    # combine prompt and neg_prompt
    prompt_embeds1_combine = torch.cat([sample_neg_prompt_embeds, prompt_embeds1], dim=0)

    return prompt_embeds1_combine
