import torch


def encode_prompts_list(pipeline, device, prompt_list):
    prompt_ids = pipeline.tokenizer(
        prompt_list,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
    return prompt_embeds


def prepare_encoded_prompts(config, accelerator, pipeline, prompt, sample_neg_prompt_embeds=None):
    if sample_neg_prompt_embeds is None:
        # generate negative prompt embeddings
        neg_prompt_embed = encode_prompts_list(pipeline, accelerator.device, [""])
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)

    prompts1 = [
        prompt
        for _ in range(config.sample.batch_size)
    ]

    # encode prompts
    prompt_embeds1 = encode_prompts_list(pipeline, accelerator.device, prompts1)
    # combine prompt and neg_prompt
    prompt_embeds1_combine = torch.cat([sample_neg_prompt_embeds, prompt_embeds1], dim=0)

    return prompt_embeds1_combine
