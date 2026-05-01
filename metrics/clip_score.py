import logging
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CLIPScoreEvaluator:
    def __init__(
            self,
            device: str = "cuda",
            torch_dtype: Optional[torch.dtype] = None,
            batch_size: int = 16,
            model_id: Optional[str] = "openai/clip-vit-large-patch14"
    ):
        if model_id is None:
            model_id = "openai/clip-vit-large-patch14"

        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device

        self.batch_size = batch_size
        self.torch_dtype = torch_dtype

        model_kwargs = {}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype

        logger.info(
            "Loading model %s on %s with dtype %s...",
            model_id,
            self.device,
            self.torch_dtype,
        )
        self.model = CLIPModel.from_pretrained(model_id, **model_kwargs).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_id)
        logger.info("Model and processor loaded successfully.")

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")
        image.load()

        if image.mode != "RGB":
            return image.convert("RGB")
        return image.copy()

    def _prepare_inputs(
            self,
            images: Sequence[Image.Image],
            img_descriptions: Sequence[str],
    ):
        prepared_images = [self._prepare_image(image) for image in images]
        inputs = self.processor(
            text=list(img_descriptions),
            images=prepared_images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    @staticmethod
    def _compute_clip_scores(
            image_embeds: torch.Tensor,
            text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = image_embeds.float()
        text_embeds = text_embeds.float()

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # CLIPScore uses cosine similarity between image and text embeddings,
        # clipped to be non-negative.
        scores = torch.sum(image_embeds * text_embeds, dim=-1)
        return torch.clamp(scores, min=0.0)

    def evaluate_batch(
            self,
            images: Sequence[Image.Image],
            img_descriptions: Sequence[str],
    ) -> List[float]:
        if len(images) != len(img_descriptions):
            raise ValueError(
                "The number of images must match the number of descriptions: "
                f"{len(images)} != {len(img_descriptions)}"
            )

        inputs = self._prepare_inputs(images, img_descriptions)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            scores = self._compute_clip_scores(outputs.image_embeds, outputs.text_embeds)

        return scores.detach().cpu().tolist()

    def evaluate_sample(self, image: Image.Image, img_description: str) -> float:
        return self.evaluate_batch([image], [img_description])[0]

    def evaluate(
            self,
            images: List[Image.Image],
            img_descriptions: List[str],
    ) -> float:
        if len(images) != len(img_descriptions):
            raise ValueError(
                "The number of images must match the number of descriptions: "
                f"{len(images)} != {len(img_descriptions)}"
            )
        if not images:
            raise ValueError("Expected at least one image to evaluate.")

        scores = []
        total = len(images)

        for i in tqdm(range(0, total, self.batch_size), desc="Evaluating CLIPScore"):
            batch_images = images[i:i + self.batch_size]
            batch_descriptions = img_descriptions[i:i + self.batch_size]
            scores.extend(self.evaluate_batch(batch_images, batch_descriptions))

        return float(np.mean(scores))


def compute_clip_score(
        images: List[Image.Image],
        prompts: List[str],
        device: str = "cuda",
        batch_size: int = 32,
        model_id: Optional[str] = "openai/clip-vit-large-patch14",
        torch_dtype: Optional[torch.dtype] = None,
) -> float:
    evaluator = CLIPScoreEvaluator(
        device=device,
        torch_dtype=torch_dtype,
        batch_size=batch_size,
        model_id=model_id,
    )
    avg_score = evaluator.evaluate(images, prompts)
    logger.info("CLIPScore computed: %.4f", avg_score)
    return avg_score
