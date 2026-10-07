from collections.abc import Mapping

import torch
import torch.nn.functional as F


ROBUST_LOW_SIGMA = 0.5
ROBUST_HIGH_SIGMA = 1.5
ROBUST_SCALE_EPS = 1e-6
HYSTERESIS_GROWTH_STEPS = 4


def _reshape_cross_attention(cross_mask):
    if cross_mask.ndim != 3:
        raise ValueError(f"Expected cross-attention shape (B, HW, items), got {tuple(cross_mask.shape)}")

    bsize, width_height, item_count = cross_mask.shape
    width = int(width_height ** 0.5)
    if width * width != width_height:
        raise ValueError(f"Cross-attention map size must be square, got HW={width_height}")

    return cross_mask.float().permute(0, 2, 1).reshape(bsize, item_count, width, width)


def fuse_cross_attention_maps(cross_masks, target_size=64):
    if isinstance(cross_masks, torch.Tensor):
        cross_masks = {int(cross_masks.shape[1] ** 0.5): cross_masks}
    if not isinstance(cross_masks, Mapping) or not cross_masks:
        raise RuntimeError("UNet did not return cross-attention maps")

    resized_maps = []
    expected_shape = None
    for _, cross_mask in sorted(cross_masks.items()):
        attention_map = _reshape_cross_attention(cross_mask)
        current_shape = attention_map.shape[:2]
        if expected_shape is None:
            expected_shape = current_shape
        elif current_shape != expected_shape:
            raise ValueError(
                "Cross-attention maps from different scales must have matching "
                f"batch and item dimensions, got {expected_shape} and {current_shape}"
            )

        if attention_map.shape[-2:] != (target_size, target_size):
            attention_map = F.interpolate(
                attention_map,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
        resized_maps.append(attention_map)

    return torch.stack(resized_maps, dim=0).mean(dim=0)


def _grow_from_confident_seeds(seeds, candidates):
    grown = seeds
    for _ in range(HYSTERESIS_GROWTH_STEPS):
        neighbours = F.max_pool2d(grown, kernel_size=3, stride=1, padding=1)
        grown = torch.maximum(grown, neighbours * candidates)
    return grown


def _close_small_holes(mask):
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return 1.0 - F.max_pool2d(1.0 - dilated, kernel_size=3, stride=1, padding=1)


def _remove_small_islands(mask):
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)


def postprocess_attention_masks(mask):
    mask = mask.float()

    # Hysteresis already suppresses most background. Remove only isolated one- or
    # two-pixel speckles so that genuinely small objects are not discarded.
    local_support = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) * 9.0
    mask = mask * (local_support >= 3.0).to(mask.dtype)
    mask = _remove_small_islands(mask)
    mask = _close_small_holes(mask)
    return (mask >= 0.5).to(mask.dtype)


def binarize_attention_maps(attention_maps, threshold_type="robust", threshold_scale=1.0):
    threshold_type = str(threshold_type).lower()
    if threshold_type == "mean":
        threshold = float(threshold_scale) * attention_maps.mean(dim=(-2, -1), keepdim=True)
        binary_maps = (attention_maps >= threshold).to(attention_maps.dtype)
    elif threshold_type == "robust":
        flat_maps = attention_maps.flatten(start_dim=-2)
        median = flat_maps.median(dim=-1, keepdim=True).values
        deviations = (flat_maps - median).abs()
        mad = deviations.median(dim=-1, keepdim=True).values
        robust_sigma = 1.4826 * mad
        standard_sigma = flat_maps.std(dim=-1, keepdim=True, unbiased=False)
        sigma = torch.where(robust_sigma > ROBUST_SCALE_EPS, robust_sigma, standard_sigma)

        median = median.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
        scale = float(threshold_scale)
        low_threshold = median + ROBUST_LOW_SIGMA * scale * sigma
        high_threshold = median + ROBUST_HIGH_SIGMA * scale * sigma

        informative = sigma > ROBUST_SCALE_EPS
        candidates = (attention_maps >= low_threshold) & informative
        seeds = (attention_maps >= high_threshold) & informative
        binary_maps = _grow_from_confident_seeds(
            seeds.to(attention_maps.dtype),
            candidates.to(attention_maps.dtype),
        )
    else:
        raise ValueError(
            f"Unknown attention mask threshold type: {threshold_type!r}. "
            "Expected 'mean' or 'robust'."
        )

    return postprocess_attention_masks(binary_maps)


def cross_attention_maps_to_object_mask(
        cross_masks,
        target_size=64,
        threshold_type="robust",
        threshold_scale=1.0,
):
    attention_maps = fuse_cross_attention_maps(cross_masks, target_size=target_size)
    if attention_maps.shape[1] == 0:
        return torch.zeros(
            attention_maps.shape[0],
            target_size,
            target_size,
            dtype=attention_maps.dtype,
            device=attention_maps.device,
        )

    object_maps = binarize_attention_maps(
        attention_maps,
        threshold_type=threshold_type,
        threshold_scale=threshold_scale,
    )
    return object_maps.amax(dim=1)
