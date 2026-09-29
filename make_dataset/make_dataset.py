import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from enum import Enum
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import datasets
from PIL import Image
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from background_masking import add_background_masks_to_dataset
from depth_masking import depth_masks_for_dataset
from sam_masking import sam_config, sam_masks_for_dataset

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


class MaskMethod(Enum):
    DINO_SAM = 'dino_sam'
    DEPTH_MAP = 'depth'
    BACKGROUND = 'background'


def count_words(prompt):
    if prompt is None:
        return 0
    return len(prompt.strip().split())


def add_masks_to_dataset(dataset, mask_method=MaskMethod.DINO_SAM, batch_size=1, hf_token=None, mask_storage="png"):
    if "mask" in dataset.column_names:
        dataset = dataset.remove_columns("mask")
    if mask_method == MaskMethod.DINO_SAM:
        masks = sam_masks_for_dataset(dataset, sam_config)
    elif mask_method == MaskMethod.DEPTH_MAP:
        masks = depth_masks_for_dataset(dataset)
    elif mask_method == MaskMethod.BACKGROUND:
        return add_background_masks_to_dataset(
            dataset,
            batch_size=batch_size,
            hf_token=hf_token,
            mask_storage=mask_storage,
        )
    else:
        raise NotImplementedError(f'Method to make object masks "{mask_method.value}" is not implemented')
    dataset_masks = datasets.Dataset.from_dict({"mask": masks})
    return datasets.concatenate_datasets([dataset, dataset_masks], axis=1)


def load_source_dataset(data_path, split="train", from_disk=False):
    if from_disk:
        dataset = load_from_disk(data_path)
        if isinstance(dataset, datasets.DatasetDict):
            if split not in dataset:
                available = ", ".join(dataset.keys())
                raise ValueError(f'Split "{split}" was not found. Available splits: {available}')
            dataset = dataset[split]
        return dataset
    return load_dataset(data_path, name=None, split=split)


def make_dataset(data_path, save_path, to_filter=True, max_samples=-1, make_masks=False,
                 mask_method=MaskMethod.DINO_SAM, batch_size=1, from_disk=False, split="train", hf_token=None,
                 recompress_images=False, image_format="jpeg", image_quality=90, mask_storage="png"):
    if data_path is None:
        return
    dataset = load_source_dataset(data_path, split=split, from_disk=from_disk)
    if ('prompt' not in dataset.column_names) and ('caption' in dataset.column_names):
        dataset = dataset.rename_column('caption', 'prompt')
    if not from_disk:
        columns_to_remove = [col for col in dataset.column_names if col not in ['image', 'prompt']]
        dataset = dataset.remove_columns(columns_to_remove)
    if to_filter:
        dataset = dataset.filter(lambda row: 0 < count_words(row['prompt']) <= 15)
    if 0 < max_samples < len(dataset):
        dataset = dataset.train_test_split(train_size=max_samples, shuffle=False, seed=1234)['train']

    if make_masks:
        dataset = add_masks_to_dataset(
            dataset,
            mask_method=mask_method,
            batch_size=batch_size,
            hf_token=hf_token,
            mask_storage=mask_storage,
        )
    if recompress_images:
        dataset = recompress_dataset_images(
            dataset,
            image_format=image_format,
            image_quality=image_quality,
            batch_size=batch_size,
        )

    dataset.save_to_disk(save_path)


def first_existing(row, names, default=None):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def to_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_coyo_prompt(row):
    return first_existing(row, ["prompt", "caption", "text", "description"], "")


def get_coyo_url(row):
    return first_existing(row, ["url", "image_url", "sample_url", "jpg"], None)


def get_coyo_score(row, names, default=None):
    return to_float(first_existing(row, names, None), default)


def get_coyo_dimensions(row):
    width = to_int(first_existing(row, ["width", "image_width", "original_width"], None), None)
    height = to_int(first_existing(row, ["height", "image_height", "original_height"], None), None)
    return width, height


def image_size_passes_filters(width, height, min_resolution=512, min_aspect_ratio=0.8, max_aspect_ratio=1.25,
                              allow_missing=True):
    if width is None or height is None:
        return allow_missing
    if width <= 0 or height <= 0:
        return False
    if width < min_resolution or height < min_resolution:
        return False
    aspect_ratio = width / height
    return min_aspect_ratio <= aspect_ratio <= max_aspect_ratio


def prepare_stable_diffusion_image(image, target_resolution=512):
    if target_resolution <= 0:
        return image

    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image.resize((target_resolution, target_resolution), RESAMPLE_LANCZOS)


def normalized_image_format(image_format):
    if image_format is None:
        return None
    image_format = image_format.lower()
    if image_format in ["none", "pil", "raw"]:
        return None
    if image_format in ["jpg", "jpeg"]:
        return "JPEG"
    if image_format == "webp":
        return "WEBP"
    if image_format == "png":
        return "PNG"
    raise ValueError(f"Unsupported image format: {image_format}")


def encode_image_for_dataset(image, image_format="jpeg", image_quality=90):
    image_format = normalized_image_format(image_format)
    if image_format is None:
        return image

    if image_format in ["JPEG", "WEBP"] and image.mode != "RGB":
        image = image.convert("RGB")

    save_kwargs = {}
    if image_format in ["JPEG", "WEBP"]:
        save_kwargs["quality"] = image_quality
    if image_format == "JPEG":
        save_kwargs["optimize"] = True
        save_kwargs["progressive"] = True
    elif image_format == "WEBP":
        save_kwargs["method"] = 6
    elif image_format == "PNG":
        save_kwargs["optimize"] = True

    image_bytes = BytesIO()
    image.save(image_bytes, format=image_format, **save_kwargs)
    return {"bytes": image_bytes.getvalue(), "path": None}


def recompress_dataset_images(dataset, image_format="jpeg", image_quality=90, batch_size=64):
    if normalized_image_format(image_format) is None:
        return dataset
    if "image" not in dataset.column_names:
        raise ValueError('Column "image" was not found in dataset')
    if "image" in dataset.features:
        dataset = dataset.cast_column("image", datasets.Image(decode=True))

    def encode_batch(batch):
        images = [
            encode_image_for_dataset(image.convert("RGB"), image_format=image_format, image_quality=image_quality)
            for image in batch["image"]
        ]
        return {"image": images}

    return dataset.map(
        encode_batch,
        batched=True,
        batch_size=batch_size,
        writer_batch_size=batch_size,
        desc=f"Recompressing images as {normalized_image_format(image_format)}",
    )


def coyo_quality_score(row):
    aesthetic = get_coyo_score(row, ["aesthetic_score_laion_v2", "aesthetic_score", "aesthetic"], 0.0)
    clip = get_coyo_score(row, ["clip_similarity_vitl14", "clip_similarity_vitb32", "clip_similarity"], 0.0)
    watermark = get_coyo_score(row, ["watermark_score", "watermark_probability"], 0.0)
    return aesthetic + 5.0 * clip - watermark


def coyo_quality_metadata(row):
    width, height = get_coyo_dimensions(row)
    return {
        "url": get_coyo_url(row),
        "quality_score": coyo_quality_score(row),
        "aesthetic_score": get_coyo_score(row, ["aesthetic_score_laion_v2", "aesthetic_score", "aesthetic"], None),
        "clip_similarity": get_coyo_score(row, ["clip_similarity_vitl14", "clip_similarity_vitb32", "clip_similarity"], None),
        "watermark_score": get_coyo_score(row, ["watermark_score", "watermark_probability"], None),
        "metadata_width": width if width is not None else -1,
        "metadata_height": height if height is not None else -1,
    }


def coyo_metadata_passes_filters(
        row,
        min_resolution=512,
        min_aspect_ratio=0.8,
        max_aspect_ratio=1.25,
        min_words=3,
        max_words=30,
        min_aesthetic=5.5,
        min_clip=0.30,
        max_watermark=0.3,
):
    prompt = get_coyo_prompt(row)
    url = get_coyo_url(row)
    if not prompt or not url:
        return False

    word_count = to_int(first_existing(row, ["word_count"], None), None)
    if word_count is None:
        word_count = count_words(prompt)
    if word_count < min_words or word_count > max_words:
        return False

    width, height = get_coyo_dimensions(row)
    if not image_size_passes_filters(
            width,
            height,
            min_resolution=min_resolution,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            allow_missing=True,
    ):
        return False

    aesthetic = get_coyo_score(row, ["aesthetic_score_laion_v2", "aesthetic_score", "aesthetic"], None)
    if aesthetic is None or aesthetic < min_aesthetic:
        return False

    clip = get_coyo_score(row, ["clip_similarity_vitl14", "clip_similarity_vitb32", "clip_similarity"], None)
    if clip is None or clip < min_clip:
        return False

    watermark = get_coyo_score(row, ["watermark_score", "watermark_probability"], None)
    if watermark is None or watermark > max_watermark:
        return False

    return True


def image_request_headers():
    return {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def download_image(url, timeout=10, max_download_bytes=None):
    request = Request(url, headers=image_request_headers())
    with urlopen(request, timeout=timeout) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and max_download_bytes is not None:
            if int(content_length) > max_download_bytes:
                raise ValueError(f"Image is too large: {content_length} bytes")

        chunks = []
        total_size = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if max_download_bytes is not None and total_size > max_download_bytes:
                raise ValueError(f"Image is too large: {total_size} bytes")
            chunks.append(chunk)
        image_bytes = b"".join(chunks)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.load()
    return image


def build_coyo_example(
        row,
        target_resolution=512,
        min_resolution=512,
        min_aspect_ratio=0.8,
        max_aspect_ratio=1.25,
        download_timeout=5,
        max_download_bytes=None,
        image_format="jpeg",
        image_quality=90,
):
    url = get_coyo_url(row)
    prompt = get_coyo_prompt(row)
    metadata = coyo_quality_metadata(row)
    try:
        image = download_image(url, timeout=download_timeout, max_download_bytes=max_download_bytes)
    except (HTTPError, URLError, OSError, TimeoutError, TypeError, ValueError):
        return None, "failed_download"

    downloaded_width = image.width
    downloaded_height = image.height
    if not image_size_passes_filters(
            downloaded_width,
            downloaded_height,
            min_resolution=min_resolution,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            allow_missing=False,
    ):
        return None, "bad_resolution"

    image = prepare_stable_diffusion_image(image, target_resolution=target_resolution)
    example = {
        "image": encode_image_for_dataset(image, image_format=image_format, image_quality=image_quality),
        "prompt": prompt,
        "downloaded_width": downloaded_width,
        "downloaded_height": downloaded_height,
    }
    example.update(metadata)
    return example, None


def coyo_dataset_features():
    return datasets.Features({
        "image": datasets.Image(),
        "prompt": datasets.Value("string"),
        "url": datasets.Value("string"),
        "quality_score": datasets.Value("float32"),
        "aesthetic_score": datasets.Value("float32"),
        "clip_similarity": datasets.Value("float32"),
        "watermark_score": datasets.Value("float32"),
        "metadata_width": datasets.Value("int32"),
        "metadata_height": datasets.Value("int32"),
        "downloaded_width": datasets.Value("int32"),
        "downloaded_height": datasets.Value("int32"),
    })


def update_coyo_quality_sums(stats, example):
    for key in ["quality_score", "aesthetic_score", "clip_similarity", "watermark_score"]:
        value = example.get(key)
        if value is None:
            continue
        stats[f"{key}_sum"] += float(value)
        stats[f"{key}_count"] += 1


def format_coyo_quality_from_stats(stats):
    def mean_for(key):
        count = stats[f"{key}_count"]
        if count == 0:
            return "n/a"
        return f"{stats[f'{key}_sum'] / count:.4f}"

    return (
        f"quality_score: {mean_for('quality_score')}, "
        f"aesthetic: {mean_for('aesthetic_score')}, "
        f"clip: {mean_for('clip_similarity')}, "
        f"watermark: {mean_for('watermark_score')}"
    )


def generate_coyo_examples_direct(
        data_path=None,
        max_samples=-1,
        quality_filter=True,
        target_resolution=512,
        min_resolution=512,
        min_aspect_ratio=0.8,
        max_aspect_ratio=1.25,
        min_words=3,
        max_words=30,
        min_aesthetic=5.5,
        min_clip=0.30,
        max_watermark=0.3,
        download_timeout=5,
        max_download_bytes=None,
        image_format="jpeg",
        image_quality=90,
        num_workers=8,
        scan_limit=-1,
):
    if data_path is None:
        data_path = "kakaobrain/coyo-700m"

    metadata_dataset = load_dataset(data_path, name=None, split="train", streaming=True)
    stats = {
        "seen": 0,
        "metadata_passed": 0,
        "attempted_downloads": 0,
        "failed_downloads": 0,
        "skipped_bad_resolution": 0,
        "saved": 0,
        "quality_score_sum": 0.0,
        "quality_score_count": 0,
        "aesthetic_score_sum": 0.0,
        "aesthetic_score_count": 0,
        "clip_similarity_sum": 0.0,
        "clip_similarity_count": 0,
        "watermark_score_sum": 0.0,
        "watermark_score_count": 0,
    }

    metadata_filter_kwargs = {
        "min_resolution": min_resolution,
        "min_aspect_ratio": min_aspect_ratio,
        "max_aspect_ratio": max_aspect_ratio,
        "min_words": min_words,
        "max_words": max_words,
        "min_aesthetic": min_aesthetic,
        "min_clip": min_clip,
        "max_watermark": max_watermark,
    }
    worker_kwargs = {
        "target_resolution": target_resolution,
        "min_resolution": min_resolution,
        "min_aspect_ratio": min_aspect_ratio,
        "max_aspect_ratio": max_aspect_ratio,
        "download_timeout": download_timeout,
        "max_download_bytes": max_download_bytes,
        "image_format": image_format,
        "image_quality": image_quality,
    }

    def row_passes_metadata(row):
        return (not quality_filter) or coyo_metadata_passes_filters(row, **metadata_filter_kwargs)

    def reached_target():
        return max_samples != -1 and stats["saved"] >= max_samples

    def handle_result(example, status):
        stats["attempted_downloads"] += 1
        if example is not None:
            if reached_target():
                return None
            stats["saved"] += 1
            update_coyo_quality_sums(stats, example)
            return example
        if status == "bad_resolution":
            stats["skipped_bad_resolution"] += 1
        else:
            stats["failed_downloads"] += 1
        return None

    def update_progress(pbar):
        target = "all" if max_samples == -1 else str(max_samples)
        pbar.set_description(
            f"{stats['saved']} / {target} saved, "
            f"{stats['attempted_downloads']} downloads tried"
        )
        pbar.set_postfix(
            metadata_passed=stats["metadata_passed"],
            failed=stats["failed_downloads"],
            bad_res=stats["skipped_bad_resolution"],
        )

    try:
        if num_workers <= 1:
            for row in (pbar := tqdm(metadata_dataset, desc="Direct COYO download")):
                stats["seen"] += 1
                if 0 < scan_limit < stats["seen"]:
                    break
                if reached_target():
                    break
                if not row_passes_metadata(row):
                    continue

                stats["metadata_passed"] += 1
                example, status = build_coyo_example(row, **worker_kwargs)
                example = handle_result(example, status)
                if stats["attempted_downloads"] % 100 == 0:
                    update_progress(pbar)
                if example is not None:
                    yield example
            return

        max_in_flight = max(1, num_workers * 2)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()

            def submit_row(row):
                futures.add(executor.submit(build_coyo_example, row, **worker_kwargs))

            def drain_finished(block=False):
                nonlocal futures
                if not futures:
                    return []
                if block:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                else:
                    done = {future for future in futures if future.done()}
                    futures -= done

                ready_examples = []
                for future in done:
                    try:
                        example, status = future.result()
                    except Exception:
                        example, status = (None, "failed_download")
                    example = handle_result(example, status)
                    if example is not None:
                        ready_examples.append(example)
                return ready_examples

            for row in (pbar := tqdm(metadata_dataset, desc="Direct COYO download")):
                stats["seen"] += 1
                if 0 < scan_limit < stats["seen"]:
                    break

                for example in drain_finished(block=False):
                    yield example
                if reached_target():
                    break
                if not row_passes_metadata(row):
                    continue

                stats["metadata_passed"] += 1
                submit_row(row)

                while len(futures) >= max_in_flight and not reached_target():
                    for example in drain_finished(block=True):
                        yield example

                if stats["metadata_passed"] % 200 == 0:
                    update_progress(pbar)

            while futures and not reached_target():
                for example in drain_finished(block=True):
                    yield example

            for future in futures:
                future.cancel()
    finally:
        print(
            "COYO processing stats. "
            f"Saved: {stats['saved']}, scanned: {stats['seen']}, "
            f"metadata passed: {stats['metadata_passed']}, "
            f"attempted downloads: {stats['attempted_downloads']}, "
            f"failed downloads: {stats['failed_downloads']}, "
            f"skipped by real resolution: {stats['skipped_bad_resolution']}."
        )
        print(f"Mean quality of saved images. {format_coyo_quality_from_stats(stats)}.")


def make_coyo_dataset(
        data_path,
        save_path,
        max_samples=3000,
        target_resolution=512,
        min_resolution=512,
        min_aspect_ratio=0.8,
        max_aspect_ratio=1.25,
        min_words=3,
        max_words=30,
        min_aesthetic=5.5,
        min_clip=0.30,
        max_watermark=0.3,
        quality_filter=True,
        download_timeout=5,
        max_download_bytes=None,
        image_format="jpeg",
        image_quality=90,
        num_workers=8,
        scan_limit=-1,
        make_masks=False,
        mask_method=MaskMethod.DINO_SAM,
        batch_size=1,
        hf_token=None,
        cache_dir=None,
        mask_storage="png",
):
    if data_path is None:
        data_path = "kakaobrain/coyo-700m"

    features = coyo_dataset_features()
    from_generator_kwargs = {
        "features": features,
        "gen_kwargs": {
            "data_path": data_path,
            "max_samples": max_samples,
            "quality_filter": quality_filter,
            "target_resolution": target_resolution,
            "min_resolution": min_resolution,
            "min_aspect_ratio": min_aspect_ratio,
            "max_aspect_ratio": max_aspect_ratio,
            "min_words": min_words,
            "max_words": max_words,
            "min_aesthetic": min_aesthetic,
            "min_clip": min_clip,
            "max_watermark": max_watermark,
            "download_timeout": download_timeout,
            "max_download_bytes": max_download_bytes,
            "image_format": image_format,
            "image_quality": image_quality,
            "num_workers": num_workers,
            "scan_limit": scan_limit,
        },
    }
    if cache_dir is not None:
        from_generator_kwargs["cache_dir"] = cache_dir

    dataset = datasets.Dataset.from_generator(generate_coyo_examples_direct, **from_generator_kwargs)

    if max_samples != -1 and len(dataset) < max_samples:
        raise RuntimeError(
            f"Only downloaded {len(dataset)} COYO samples out of requested {max_samples}. "
            f"Increase --coyo_scan_limit or relax quality filters."
        )
    if len(dataset) == 0:
        raise RuntimeError(
            "No COYO samples were downloaded. "
            "Increase --coyo_scan_limit, relax filters, or check network access to image URLs."
        )
    if make_masks:
        dataset = add_masks_to_dataset(
            dataset,
            mask_method=mask_method,
            batch_size=batch_size,
            hf_token=hf_token,
            mask_storage=mask_storage,
        )
    dataset.save_to_disk(save_path)
    print(
        f"Saved {len(dataset)} COYO samples to {save_path}. "
        f"Images are center-cropped and resized to {target_resolution}x{target_resolution}."
    )


# make_dataset('poloclub/diffusiondb', '2m_first_10k', '/home/ergrishina_2/Diploma/diffusiondb')
# make_dataset('mlx-community/dreambooth-dog6', None, '/home/ergrishina_2/Diploma/dog6')
# make_dataset('Mercity/laion-subset', None, '/home/ergrishina_2/Diploma/laion', to_filter=False)
# make_dataset('Mercity/laion-subset', None, '/home/ergrishina_2/Diploma/laion_3k',
#              to_filter=False, max_samples=3000)
# make_dataset('Mercity/laion-subset', None, '/home/ergrishina_2/Diploma/laion_3k_masks',
#              to_filter=False, max_samples=3000, make_masks=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing arguments from console")

    parser.add_argument("--dataset_type", type=str, default="generic", choices=["generic", "coyo"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--from_disk", type=int, default=0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--to_filter", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--make_masks", type=int, default=0)
    parser.add_argument("--mask_method", type=str, default='dino_sam')
    parser.add_argument("--mask_storage", type=str, default="png", choices=["png", "array", "float32"])
    parser.add_argument("--batch_size", "--bs", type=int, default=1)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--recompress_images", type=int, default=0)
    parser.add_argument("--image_format", type=str, default="jpeg", choices=["jpeg", "jpg", "webp", "png", "none", "pil", "raw"])
    parser.add_argument("--image_quality", type=int, default=95)
    parser.add_argument("--coyo_target_resolution", type=int, default=512)
    parser.add_argument("--coyo_min_resolution", type=int, default=512)
    parser.add_argument("--coyo_min_aspect_ratio", type=float, default=0.8)
    parser.add_argument("--coyo_max_aspect_ratio", type=float, default=1.25)
    parser.add_argument("--coyo_min_words", type=int, default=3)
    parser.add_argument("--coyo_max_words", type=int, default=30)
    parser.add_argument("--coyo_min_aesthetic", type=float, default=5.0)
    parser.add_argument("--coyo_min_clip", type=float, default=0.30)
    parser.add_argument("--coyo_max_watermark", type=float, default=0.3)
    parser.add_argument("--coyo_quality_filter", type=int, default=1)
    parser.add_argument("--coyo_download_timeout", type=float, default=5)
    parser.add_argument("--coyo_max_download_mb", type=float, default=30.0)
    parser.add_argument("--coyo_num_workers", type=int, default=8)
    parser.add_argument("--coyo_scan_limit", type=int, default=1000000)
    parser.add_argument("--coyo_cache_dir", type=str, default=None)

    args = parser.parse_args()
    if args.mask_method == 'dino_sam':
        mask_method = MaskMethod.DINO_SAM
    elif args.mask_method == 'depth':
        mask_method = MaskMethod.DEPTH_MAP
    elif args.mask_method in ['background', 'bg']:
        mask_method = MaskMethod.BACKGROUND
    else:
        raise ValueError(f'Unknown method to make object masks: {args.mask_method}')

    coyo_max_download_bytes = None
    if args.coyo_max_download_mb is not None and args.coyo_max_download_mb > 0:
        coyo_max_download_bytes = int(args.coyo_max_download_mb * 1024 * 1024)

    if args.dataset_type == "coyo":
        make_coyo_dataset(
            args.data_path,
            args.save_path,
            max_samples=args.max_samples,
            target_resolution=args.coyo_target_resolution,
            min_resolution=args.coyo_min_resolution,
            min_aspect_ratio=args.coyo_min_aspect_ratio,
            max_aspect_ratio=args.coyo_max_aspect_ratio,
            min_words=args.coyo_min_words,
            max_words=args.coyo_max_words,
            min_aesthetic=args.coyo_min_aesthetic,
            min_clip=args.coyo_min_clip,
            max_watermark=args.coyo_max_watermark,
            quality_filter=bool(args.coyo_quality_filter),
            download_timeout=args.coyo_download_timeout,
            max_download_bytes=coyo_max_download_bytes,
            image_format=args.image_format,
            image_quality=args.image_quality,
            num_workers=args.coyo_num_workers,
            scan_limit=args.coyo_scan_limit,
            make_masks=bool(args.make_masks),
            mask_method=mask_method,
            batch_size=args.batch_size,
            hf_token=args.hf_token,
            cache_dir=args.coyo_cache_dir,
            mask_storage=args.mask_storage,
        )
    else:
        make_dataset(args.data_path, args.save_path, args.to_filter, args.max_samples,
                     bool(args.make_masks), mask_method, args.batch_size,
                     from_disk=bool(args.from_disk), split=args.split, hf_token=args.hf_token,
                     recompress_images=bool(args.recompress_images),
                     image_format=args.image_format,
                     image_quality=args.image_quality,
                     mask_storage=args.mask_storage)
