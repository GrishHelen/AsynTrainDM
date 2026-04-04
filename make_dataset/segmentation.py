from dataclasses import dataclass
from typing import List

import numpy as np
import spacy
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


@dataclass
class MaskingConfig:
    grounding_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_model_id: str = "facebook/sam-vit-base"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    sam_iou_threshold: float = 0.85
    max_phrases: int = 4
    max_boxes: int = 8
    fallback_mode: str = "empty"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def extract_objects(prompt: str, max_phrases: int) -> List[str]:
    try:
        doc = nlp(prompt)
    except Exception as e:
        print(f'ERROR, prompt: {prompt}')
    objects = [chunk.text for chunk in doc.noun_chunks]
    if max_phrases > 0:
        objects = objects[:max_phrases]
    return objects


class PromptMaskAnnotator:
    def __init__(self, config: MaskingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.gd_processor = AutoProcessor.from_pretrained(config.grounding_model_id)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(config.grounding_model_id).to(self.device)
        self.gd_model.eval()

        self.sam_processor = SamProcessor.from_pretrained(config.sam_model_id)
        self.sam_model = SamModel.from_pretrained(config.sam_model_id).to(self.device)
        self.sam_model.eval()

    def _predict_boxes(self, image: Image.Image, objects: List[str]) -> torch.Tensor:
        # Grounding DINO expects text input aligned with the image batch.
        width, height = image.size
        text = '. '.join(objects) + '.'
        gd_inputs = self.gd_processor(images=image, text=text, return_tensors="pt").to(self.device)

        gd_outputs = self.gd_model(**gd_inputs)
        result = self.gd_processor.post_process_grounded_object_detection(
            gd_outputs,
            gd_inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        boxes = result.get("boxes", torch.tensor([[0., 0., width, height]], dtype=torch.float32))
        box_scores = result.get("scores", torch.zeros((1,)))

        if len(boxes) > self.config.max_boxes:
            order = torch.argsort(box_scores, descending=True)[: self.config.max_boxes]
            boxes = boxes[order]

        if len(boxes) == 0:
            boxes = torch.tensor([[0., 0., width, height]], dtype=torch.float32)

        return boxes

    def _predict_mask_from_boxes(self, image: Image.Image, boxes: torch.Tensor) -> torch.Tensor:
        sam_inputs = self.sam_processor(
            images=image,
            input_boxes=[boxes.tolist()],
            return_tensors="pt",
        )
        sam_outputs = self.sam_model(**sam_inputs, multimask_output=False)

        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]
        iou_scores = sam_outputs.iou_scores[0].detach().cpu()

        # shape to [num_boxes, H, W]
        if masks.ndim == 4:
            masks = masks[:, 0]
            iou_scores = iou_scores[:, 0]
        masks = masks.float()

        keep = iou_scores >= self.config.sam_iou_threshold
        if keep.any():
            final_mask = masks[keep].any(dim=0)
        else:
            best = torch.argmax(iou_scores).item()
            final_mask = masks[best]

        # final_mask: (image.height, image.width)
        return final_mask.to(dtype=torch.float32)

    @torch.inference_mode()
    def predict_mask(self, image: Image.Image, prompt: str) -> torch.Tensor:
        objects = extract_objects(prompt, max_phrases=self.config.max_phrases)
        if len(objects) == 0:
            return self._fallback_mask(image)

        boxes = self._predict_boxes(image, objects)
        mask = self._predict_mask_from_boxes(image, boxes)

        return mask

    def _fallback_mask(self, image: Image.Image) -> torch.Tensor:
        if self.config.fallback_mode == "full":
            return torch.ones((image.height, image.width), dtype=torch.float32)
        return torch.zeros((image.height, image.width), dtype=torch.float32)


def masks_for_dataset(
        dataset,
        config: MaskingConfig,
):
    annotator = PromptMaskAnnotator(config)
    masks = []

    for idx in tqdm(range(len(dataset)), desc="Precomputing masks"):
        item = dataset[idx]
        image = item["image"].convert("RGB")
        prompt = item.get("prompt", "")

        mask = annotator.predict_mask(image, prompt)
        masks.append(mask)

    return masks


nlp = spacy.load("en_core_web_sm")

config = MaskingConfig(
    grounding_model_id="IDEA-Research/grounding-dino-tiny",
    sam_model_id="facebook/sam-vit-base",
    box_threshold=0.35,
    text_threshold=0.25,
    sam_iou_threshold=0.85,
    max_phrases=-1,
    max_boxes=8,
    fallback_mode="full",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
