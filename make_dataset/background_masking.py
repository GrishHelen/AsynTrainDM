import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

os.environ['HF_TOKEN'] = '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'


class BackgroundMasking:
    def __init__(self, model_id='briaai/RMBG-2.0', device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
        self.model = self.model.eval().to(device)

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


def background_masks_for_dataset(dataset, batch_size=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask_predictor = BackgroundMasking(device=device)
    all_masks = []

    for idx in tqdm(range(0, len(dataset), batch_size), desc="Precomputing masks"):
        loc_n = min(batch_size, len(dataset) - idx)
        images = [dataset[idx + i]["image"].convert("RGB") for i in range(loc_n)]
        pred_masks = mask_predictor.predict_mask(images)
        all_masks.extend(pred_masks)

    return all_masks
