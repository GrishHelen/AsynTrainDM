import os
from io import BytesIO

import datasets
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
from huggingface_hub import login


class BackgroundMasking:
    def __init__(self, model_id='briaai/RMBG-2.0', device="cuda", hf_token=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
        if hf_token:
            login(token=hf_token)
            load_kwargs["token"] = hf_token
        self.model = AutoModelForImageSegmentation.from_pretrained(model_id, **load_kwargs)
        self.model = self.model.eval().to(self.device)

        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def predict_mask(self, images: list[Image.Image]):
        input_images = [self.transform_image(image) for image in images]
        input_images = torch.stack(input_images, dim=0).to(self.device)
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()

        masks = [F.interpolate(pred.unsqueeze(dim=0), images[i].size[::-1]) for i, pred in enumerate(preds)]
        masks = [(mask.squeeze() > 0.5).float() for mask in masks]
        return masks


def background_masks_for_dataset(dataset, batch_size=1, hf_token=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask_predictor = BackgroundMasking(device=device, hf_token=hf_token)
    all_masks = []

    for idx in tqdm(range(0, len(dataset), batch_size), desc="Precomputing masks"):
        loc_n = min(batch_size, len(dataset) - idx)
        images = [dataset[idx + i]["image"].convert("RGB") for i in range(loc_n)]
        pred_masks = mask_predictor.predict_mask(images)
        all_masks.extend(pred_masks)

    return all_masks


def encode_binary_mask_png(mask):
    mask_array = (mask.cpu().numpy().astype("uint8") * 255)
    mask_image = Image.fromarray(mask_array, mode="L")
    mask_bytes = BytesIO()
    mask_image.save(mask_bytes, format="PNG", optimize=True)
    return {"bytes": mask_bytes.getvalue(), "path": None}


def add_background_masks_to_dataset(dataset, batch_size=1, hf_token=None, mask_storage="png"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask_predictor = BackgroundMasking(device=device, hf_token=hf_token)
    mask_storage = mask_storage.lower()
    if mask_storage not in ["png", "array", "float32"]:
        raise ValueError(f"Unsupported mask storage: {mask_storage}")

    def predict_batch(batch):
        images = [image.convert("RGB") for image in batch["image"]]
        pred_masks = mask_predictor.predict_mask(images)
        if mask_storage == "png":
            return {"mask": [encode_binary_mask_png(mask) for mask in pred_masks]}
        if mask_storage == "array":
            return {"mask": [mask.to(torch.uint8).cpu().numpy() for mask in pred_masks]}
        return {"mask": [mask.cpu().numpy() for mask in pred_masks]}

    map_kwargs = {}
    if mask_storage == "png":
        features = dataset.features.copy()
        features["mask"] = datasets.Image()
        map_kwargs["features"] = features

    return dataset.map(
        predict_batch,
        batched=True,
        batch_size=batch_size,
        writer_batch_size=batch_size,
        desc="Precomputing background masks",
        **map_kwargs,
    )
