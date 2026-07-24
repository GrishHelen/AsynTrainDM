import argparse
import logging
from pathlib import Path

from transformers import AutoProcessor, CLIPModel

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAVE_DIR = "/home/ergrishina_2/clip-vit-large-patch14"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_with_snapshot(model_id: str, save_dir: Path) -> Path:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub.snapshot_download is not available.")

    logger.info("Downloading '%s' to '%s' using snapshot_download...", model_id, save_dir)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(save_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return save_dir


def download_with_transformers(model_id: str, save_dir: Path) -> Path:
    logger.info("Downloading processor for '%s'...", model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    logger.info("Downloading model for '%s'...", model_id)
    model = CLIPModel.from_pretrained(model_id)

    logger.info("Saving processor to '%s'...", save_dir)
    processor.save_pretrained(save_dir)

    logger.info("Saving model to '%s'...", save_dir)
    try:
        model.save_pretrained(save_dir, safe_serialization=True)
    except Exception as error:
        logger.warning(
            "Could not save with safetensors (%s). Falling back to PyTorch format.",
            error,
        )
        model.save_pretrained(save_dir, safe_serialization=False)

    return save_dir


def presave_clip_model(
    model_id: str = DEFAULT_MODEL_ID,
    save_dir: Path = DEFAULT_SAVE_DIR,
    force_download: bool = False,
) -> Path:
    save_dir = Path(save_dir).resolve()

    if save_dir.exists() and any(save_dir.iterdir()) and not force_download:
        logger.info(
            "Directory '%s' is already populated. Reusing existing files.",
            save_dir,
        )
        return save_dir

    save_dir.mkdir(parents=True, exist_ok=True)

    if snapshot_download is not None:
        try:
            return download_with_snapshot(model_id, save_dir)
        except Exception as error:
            logger.warning(
                "snapshot_download failed (%s). Falling back to save_pretrained.",
                error,
            )

    return download_with_transformers(model_id, save_dir)



def main() -> None:
    saved_path = presave_clip_model()

    logger.info("Model is available at: %s", saved_path)
    logger.info(
        "Use this path later as model_id, for example: model_id='%s'",
        saved_path,
    )


if __name__ == "__main__":
    main()