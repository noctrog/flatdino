from pathlib import Path
import tempfile
from urllib.parse import urlsplit

from flatdino.data import MODELS_CACHE_ROOT, HASH_FOLDER
from flatdino.data.utils import download_file_from_url, hash_file

URL = "https://huggingface.co/nyu-visionx/RAE-collections/resolve/main/decoders/dinov2/wReg_base/ViTXL_n08/model.pt?download=true"
MODEL_NAME = "rae_decoder_vitxl"
DEFAULT_MODEL_DIR = MODELS_CACHE_ROOT / MODEL_NAME
MODEL_HASH_FILE = HASH_FOLDER / f"{MODEL_NAME}.txt"

_url_path = urlsplit(URL).path or URL
MODEL_FILENAME = Path(_url_path).name or "model.pt"


def _get_model_path(root: Path | None = None) -> Path:
    """Return the local path where the decoder checkpoint should live."""
    base_dir = Path(root) if root is not None else DEFAULT_MODEL_DIR
    return base_dir / MODEL_FILENAME


def _is_model_healthy(model_path: Path) -> bool:
    """Check whether the cached checkpoint matches the stored hash."""
    if not model_path.is_file() or not MODEL_HASH_FILE.is_file():
        return False

    try:
        stored_hash = MODEL_HASH_FILE.read_text().strip()
    except OSError:
        return False

    if not stored_hash:
        return False

    try:
        current_hash = hash_file(model_path)
    except (OSError, AssertionError):
        return False

    return current_hash == stored_hash


def download_rae_decoder(force_download: bool = False, root: Path | None = None) -> Path:
    """Ensure the PyTorch checkpoint for the RAE decoder is available locally.

    Args:
        force_download: Re-download the checkpoint even if the local copy looks valid.
        root: Optional directory override for the local cache.

    Returns:
        Path to the cached checkpoint on disk.
    """
    model_path = _get_model_path(root)

    if not force_download and _is_model_healthy(model_path):
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=model_path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_file_from_url(URL, tmp_path)
        tmp_path.replace(model_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    HASH_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_HASH_FILE.write_text(hash_file(model_path))

    return model_path


__all__ = ["download_rae_decoder", "MODEL_FILENAME", "MODEL_NAME", "URL"]


if __name__ == "__main__":
    import torch

    model_path = download_rae_decoder()
    model_pt = torch.load(model_path)
    print(model_pt.keys())
