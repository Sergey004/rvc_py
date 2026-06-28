"""Automatic model downloader for RMVPE and HuBERT from HuggingFace.

Путь для сохранения моделей определяется в порядке приоритета:
  1. Переменная окружения RVC_MODELS_DIR
  2. ~/.cache/rvc/models  (XDG-совместимый дефолт)

Пример:
    RVC_MODELS_DIR=/data/rvc_models python xiaozhi_server.py
"""
from __future__ import annotations

import os
import requests

MODEL_URLS = {
    "rmvpe.pt":      "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    "rmvpe.onnx":    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx",
    "hubert_base.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
}


def get_models_dir() -> str:
    """
    Возвращает директорию для хранения весовых файлов RVC.

    Приоритет:
      1. RVC_MODELS_DIR из окружения
      2. ~/.cache/rvc/models
    """
    env = os.environ.get("RVC_MODELS_DIR")
    if env:
        path = os.path.expanduser(env)
    else:
        path = os.path.join(os.path.expanduser("~"), ".cache", "rvc", "models")
    os.makedirs(path, exist_ok=True)
    return path


def download_model(model_name: str, out_dir: str | None = None) -> str:
    """Download a model file if not already present.

    Args:
        model_name: Key in MODEL_URLS (e.g. 'hubert_base.pt').
        out_dir: Директория для сохранения. Если None — использует get_models_dir().

    Returns:
        Абсолютный путь к скачанному файлу.
    """
    if out_dir is None:
        out_dir = get_models_dir()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, model_name)

    if os.path.exists(out_path):
        print(f"[RVC] {model_name} already exists: {out_path}")
        return out_path

    if model_name not in MODEL_URLS:
        raise ValueError(f"[RVC] Unknown model: {model_name}. Available: {list(MODEL_URLS)}")

    url = MODEL_URLS[model_name]
    print(f"[RVC] Downloading {model_name} → {out_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[RVC] Downloaded: {out_path}")
    return out_path


def download_all_models(out_dir: str | None = None) -> None:
    """Download all known models."""
    target = out_dir or get_models_dir()
    print(f"[RVC] Downloading all models to: {target}")
    for name in MODEL_URLS:
        download_model(name, out_dir=target)


if __name__ == "__main__":
    download_all_models()
