import logging
import os
import re
import sys
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QwenScoreEvaluator:
    def __init__(
            self,
            device: str = "cuda",
            torch_dtype: Optional[torch.dtype] = None,
            max_new_tokens: int = 5,
    ):
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype

        model_kwargs = {}
        if self.device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"

        logger.info(
            "Loading model %s on %s with dtype %s...",
            model_id,
            self.device,
            self.torch_dtype,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        logger.info("Model and processor loaded successfully.")

    @staticmethod
    def _build_prompt(img_description: str) -> str:
        return (
            "You are given an image and a description. "
            "Please evaluate how well the image matches the description on a scale from 0 to 9, "
            "where 0 means completely unrelated and 9 means perfectly aligned. "
            "Return only the score as a single integer without explanation.\n"
            f"Description: {img_description}"
        )

    @staticmethod
    def _parse_score(response: str, img_description: Optional[str] = None) -> int:
        match = re.search(r"\b([0-9])\b", response.strip())
        if not match:
            raise ValueError(f"Could not parse score from response: '{response}'. Prompt: {img_description}")
        return int(match.group(1))

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")
        image.load()

        if image.mode != "RGB":
            return image.convert("RGB")
        return image.copy()

    def _make_texts(self, images: Sequence[Image.Image], img_descriptions: Sequence[str]) -> List[str]:
        texts = []
        for img, desc in zip(images, img_descriptions):
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": self._build_prompt(desc)},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)
        return texts

    def _prepare_inputs(
            self,
            images: Sequence[Image.Image],
            img_descriptions: Sequence[str],
    ):
        prepared_images = [self._prepare_image(image) for image in images]
        inputs = self.processor(
            text=self._make_texts(images, img_descriptions),
            images=prepared_images,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    def evaluate_sample(self, image: Image.Image, img_description: str) -> int:
        inputs = self._prepare_inputs([image], [img_description])

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False  # Deterministic output for reproducibility
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._parse_score(response, img_description)

    def evaluate(
            self,
            images: List[Image.Image],
            img_descriptions: List[str],
    ) -> float:
        scores = []
        total = len(images)

        for i in tqdm(range(total), desc="Evaluating QwenScore"):
            score = self.evaluate_sample(images[i], img_descriptions[i])
            scores.append(score)

        return float(np.mean(scores))


def compute_qwen_score(
        images: List[Image.Image],
        prompts: List[str],
        device: str = "cuda",
        max_new_tokens: int = 5,
) -> float:
    evaluator = QwenScoreEvaluator(
        device=device,
        max_new_tokens=max_new_tokens,
    )
    avg_score = evaluator.evaluate(images, prompts)
    logger.info("QwenScore computed: %.2f", avg_score)
    return avg_score
