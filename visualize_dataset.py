import argparse
from io import BytesIO
from pathlib import Path

import datasets
from datasets import load_dataset, load_from_disk
from PIL import Image, ImageDraw, ImageFont


DEFAULT_IMAGE_COLUMNS = ("image", "jpg", "png")
DEFAULT_PROMPT_COLUMNS = ("prompt", "caption", "text", "description")

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


def load_any_dataset(dataset_path, split="train", dataset_name=None):
    path = Path(dataset_path)
    if path.exists():
        dataset = load_from_disk(str(path))
        if isinstance(dataset, datasets.DatasetDict):
            if split not in dataset:
                available = ", ".join(dataset.keys())
                raise ValueError(f'Split "{split}" was not found. Available splits: {available}')
            dataset = dataset[split]
        return dataset

    return load_dataset(dataset_path, name=dataset_name, split=split)


def choose_column(column_names, requested, candidates, kind):
    if requested is not None:
        if requested not in column_names:
            raise ValueError(f'{kind} column "{requested}" was not found. Columns: {column_names}')
        return requested

    for column in candidates:
        if column in column_names:
            return column
    raise ValueError(f"Could not infer {kind} column. Columns: {column_names}")


def as_pil_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)}")


def load_font(font_size=18):
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_text(text, draw, font, max_width):
    words = str(text).replace("\n", " ").split()
    if not words:
        return [""]

    lines = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if text_width(draw, candidate, font) <= max_width:
            line = candidate
            continue
        if line:
            lines.append(line)
        line = word
    if line:
        lines.append(line)
    return lines

def render_image_with_prompt(image, prompt, font, max_prompt_chars=400, max_lines=6, padding=12):
    prompt = str(prompt)
    if len(prompt) > max_prompt_chars:
        prompt = prompt[: max_prompt_chars - 3].rstrip() + "..."

    draw_probe = ImageDraw.Draw(Image.new("RGB", (image.width, 1)))
    lines = wrap_text(prompt, draw_probe, font, image.width - 2 * padding)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + "..."

    line_bbox = draw_probe.textbbox((0, 0), "Ag", font=font)
    line_height = line_bbox[3] - line_bbox[1] + 5
    caption_height = 2 * padding + line_height * len(lines)

    canvas = Image.new("RGB", (image.width, image.height + caption_height), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = image.height + padding
    for line in lines:
        draw.text((padding, y), line, fill="black", font=font)
        y += line_height
    return canvas


def make_output_dir(output_dir, overwrite=False):
    output_dir = Path(output_dir)
    if overwrite:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return output_dir

    for idx in range(1, 10000):
        candidate = output_dir.with_name(f"{output_dir.name}_{idx}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not create a unique output directory near {output_dir}")


def save_dataset_preview(
        dataset,
        output_dir,
        num_images,
        image_column=None,
        prompt_column=None,
        font_size=18,
        overwrite=False,
):
    image_column = choose_column(dataset.column_names, image_column, DEFAULT_IMAGE_COLUMNS, "image")
    prompt_column = choose_column(dataset.column_names, prompt_column, DEFAULT_PROMPT_COLUMNS, "prompt")
    output_dir = make_output_dir(output_dir, overwrite=overwrite)
    font = load_font(font_size=font_size)

    manifest_lines = ["idx\tfile\tprompt\n"]
    total = min(num_images, len(dataset))
    for idx in range(total):
        row = dataset[idx]
        image = as_pil_image(row[image_column])
        prompt = row[prompt_column]
        preview = render_image_with_prompt(image, prompt, font)

        filename = f"{idx:04d}.png"
        preview.save(output_dir / filename)
        prompt_clean = str(prompt).replace("\t", " ").replace("\n", " ")
        manifest_lines.append(f"{idx}\t{filename}\t{prompt_clean}\n")

    (output_dir / "prompts.tsv").write_text("".join(manifest_lines), encoding="utf-8")
    return output_dir, total, image_column, prompt_column


def parse_args():
    parser = argparse.ArgumentParser(description="Save first N dataset images with prompt captions.")
    parser.add_argument("--dataset_path", "--dataset", required=True, type=str)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", "--out", type=str, default="dataset_preview")
    parser.add_argument("--num_images", "-n", type=int, default=16)
    parser.add_argument("--image_column", type=str, default=None)
    parser.add_argument("--prompt_column", type=str, default=None)
    parser.add_argument("--font_size", type=int, default=18)
    parser.add_argument("--overwrite", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_any_dataset(args.dataset_path, split=args.split, dataset_name=args.dataset_name)
    output_dir, total, image_column, prompt_column = save_dataset_preview(
        dataset,
        output_dir=args.output_dir,
        num_images=args.num_images,
        image_column=args.image_column,
        prompt_column=args.prompt_column,
        font_size=args.font_size,
        overwrite=bool(args.overwrite),
    )
    print(
        f'Saved {total} images to "{output_dir}". '
        f'Image column: "{image_column}", prompt column: "{prompt_column}".'
    )


if __name__ == "__main__":
    main()
