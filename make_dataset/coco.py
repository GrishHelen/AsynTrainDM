from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import datasets
import numpy as np
from PIL import Image
from tqdm import tqdm

COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Skip download, file already exists: {destination}")
        return destination
    print(f"Downloading {url} -> {destination}")
    with urlopen(url) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)
    return destination


def extract_zip(archive_path: Path, extract_to: Path, expected_path: Path | None = None) -> None:
    if expected_path is not None and expected_path.exists():
        print(f"Skip extract, folder already exists: {expected_path}")
        return
    print(f"Extracting {archive_path} -> {extract_to}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_to)


def ensure_coco_dataset(root: str | Path, split: str = "train2017") -> dict[str, Path]:
    if split not in {"train2017", "val2017"}:
        raise ValueError("split must be 'train2017' or 'val2017'")
    root = Path(root)
    archives_dir = root / "archives"
    images_archive = download_file(COCO_URLS[split], archives_dir / f"{split}.zip")
    annotations_archive = download_file(
        COCO_URLS["annotations"],
        archives_dir / "annotations_trainval2017.zip",
    )
    extract_zip(images_archive, root, expected_path=root / split)
    extract_zip(
        annotations_archive,
        root,
        expected_path=root / "annotations" / f"captions_{split}.json",
    )
    image_dir = root / split
    instance_annotation_file = root / "annotations" / f"instances_{split}.json"
    caption_annotation_file = root / "annotations" / f"captions_{split}.json"
    if not image_dir.exists():
        raise FileNotFoundError(f"COCO images folder was not found: {image_dir}")
    if not instance_annotation_file.exists():
        raise FileNotFoundError(f"COCO annotation file was not found: {instance_annotation_file}")
    if not caption_annotation_file.exists():
        raise FileNotFoundError(f"COCO caption annotation file was not found: {caption_annotation_file}")
    return {
        "root": root,
        "image_dir": image_dir,
        "instance_annotation_file": instance_annotation_file,
        "caption_annotation_file": caption_annotation_file,
    }


def load_coco_api(annotation_file: str | Path):
    try:
        from pycocotools.coco import COCO
    except ImportError as error:
        raise ImportError(
            "pycocotools is required. Install it with: pip install pycocotools"
        ) from error
    return COCO(str(annotation_file))


def choose_prompt(caption_annotations: list[dict[str, Any]]) -> str:
    if not caption_annotations:
        return ""
    caption_annotations = sorted(caption_annotations, key=lambda annotation: annotation["id"])
    return caption_annotations[0]["caption"].strip()


def build_combined_mask(coco, image_id: int, image_size: tuple[int, int]) -> np.ndarray:
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    if not ann_ids:
        return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    annotations = coco.loadAnns(ann_ids)
    masks = [coco.annToMask(annotation).astype(np.uint8) for annotation in annotations]
    if not masks:
        return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    return np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8)


def generate_coco_examples(
        root: str | Path,
        split: str = "train2017",
        max_samples: int = 3000,
):
    paths = ensure_coco_dataset(root, split)
    instances_coco = load_coco_api(paths["instance_annotation_file"])
    captions_coco = load_coco_api(paths["caption_annotation_file"])

    image_ids = sorted(set(instances_coco.getImgIds()) & set(captions_coco.getImgIds()))
    collected = 0

    for image_id in tqdm(image_ids, desc=f"Building COCO subset from {split}"):
        image_info = instances_coco.loadImgs([image_id])[0]
        image_path = paths["image_dir"] / image_info["file_name"]

        caption_ann_ids = captions_coco.getAnnIds(imgIds=[image_id])
        caption_annotations = captions_coco.loadAnns(caption_ann_ids)
        prompt = choose_prompt(caption_annotations)
        if not prompt:
            continue

        mask = build_combined_mask(
            instances_coco,
            image_id=image_id,
            image_size=(image_info["width"], image_info["height"]),
        )
        if mask.max() == 0:
            continue

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            image.load()

        yield {
            "image": image,
            "mask": mask.astype(int),
            "prompt": prompt,
        }

        collected += 1
        if 0 < max_samples <= collected:
            return


def build_coco_dataset(
        root: str | Path,
        save_path: str | Path = "coco_3k",
        split: str = "train2017",
        max_samples: int = 3000,
) -> datasets.Dataset:
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "mask": datasets.Image(),
            "prompt": datasets.Value("string"),
        }
    )
    dataset = datasets.Dataset.from_generator(
        generate_coco_examples,
        features=features,
        gen_kwargs={
            "root": root,
            "split": split,
            "max_samples": max_samples,
        },
    )
    if 0 < max_samples and len(dataset) < max_samples:
        raise ValueError(f"Only collected {len(dataset)} samples, expected {max_samples}")

    save_path = Path(save_path)
    if save_path.exists():
        raise FileExistsError(f"Output path already exists: {save_path}")
    dataset.save_to_disk(str(save_path))
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and use COCO instance masks")
    parser.add_argument("--root", type=str, default="COCO")
    parser.add_argument("--split", type=str, default="train2017")
    parser.add_argument("--build_dataset", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="coco_3k")
    parser.add_argument("--max_samples", type=int, default=3000)
    args = parser.parse_args()

    if args.build_dataset:
        dataset = build_coco_dataset(
            root=args.root,
            save_path=args.save_path,
            split=args.split,
            max_samples=args.max_samples,
        )
        print(f"Saved dataset with {len(dataset)} samples to: {args.save_path}")
        return


if __name__ == "__main__":
    main()
