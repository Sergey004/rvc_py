"""Automatic model downloader for RMVPE and HuBERT from HuggingFace."""

from __future__ import annotations

import os

import requests

MODEL_URLS = {
    "rmvpe.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    "rmvpe.onnx": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx",
    "hubert_base.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
}


def download_model(model_name: str, out_dir: str = "models") -> str:
    """Download a model file if not already present.

    Args:
        model_name: Key in MODEL_URLS.
        out_dir: Directory to save the model.

    Returns:
        Path to the downloaded model file.
    """
    os.makedirs(out_dir, exist_ok=True)
    url = MODEL_URLS[model_name]
    out_path = os.path.join(out_dir, model_name)
    if os.path.exists(out_path):
        print(f"[RVC] {model_name} already exists: {out_path}")
        return out_path
    print(f"[RVC] Downloading {model_name}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[RVC] Downloaded: {out_path}")
    return out_path


def download_all_models(out_dir: str = "models") -> None:
    """Download all known models."""
    for name in MODEL_URLS:
        download_model(name, out_dir=out_dir)


if __name__ == "__main__":
    download_all_models()
