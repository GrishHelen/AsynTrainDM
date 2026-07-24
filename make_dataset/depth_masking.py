import os
import sys

import numpy as np
import torch

cur_dir = os.path.curdir
parent_dir = os.path.abspath(os.path.join(cur_dir, ".."))

# Add it to the system path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    DepthAnything3 = None
from tqdm import tqdm


class DA3MaskEstimator:
    def __init__(self, model_id="depth-anything/da3-small", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if DepthAnything3 is None:
            raise ImportError("ModuleNotFoundError: No module named 'depth_anything_3'")
        self.model = DepthAnything3.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_masks(self, images, depth_quantile=0.3):
        depths, _ = self._predict_depths_confs(images)

        masks = self._build_masks(
            depths,
            depth_quantile=depth_quantile,
        )

        return masks, depths

    def _predict_depths_confs(self, images):
        predictions = self.model.inference(images)
        depths = predictions.depth  # [N, H, W]
        confs = predictions.conf  # [N, H, W]
        depths_min = np.min(depths, axis=(1, 2))
        depths_max = np.max(depths, axis=(1, 2))
        depths = (depths - depths_min) / (depths_max - depths_min + 1e-8)
        return depths, confs

    @staticmethod
    def _build_masks(depths, depth_quantile=0.3):
        thresh = np.quantile(depths, depth_quantile, axis=(1,2), keepdims=True)

        mask = (depths < thresh).astype(np.float32)

        return mask


def depth_masks_for_dataset(dataset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask_predictor = DA3MaskEstimator(device=device)
    masks = []
    depths = []

    for idx in tqdm(range(len(dataset)), desc="Precomputing masks"):
        image = dataset[idx]["image"].convert("RGB")
        mask, depth = mask_predictor.predict_masks([image])
        masks.append(mask[0])
        depths.append(depth[0])

    torch.save(depths, '/home/ergrishina_2/Diploma/depths.pt')

    return masks
