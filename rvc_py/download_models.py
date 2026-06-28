"""Automatic model downloader for RVC via huggingface_hub.

Модели скачиваются через HF hub в стандартный кеш (~/.cache/huggingface/hub),
затем симлинкуются/копируются в RVC_MODELS_DIR.

Путь RVC_MODELS_DIR определяется в порядке приоритета:
  1. Переменная окружения RVC_MODELS_DIR
  2. ~/.cache/rvc/models

Переменные окружения huggingface_hub тоже работают:
  HF_TOKEN        — для приватных репо
  HF_HOME         — альтернативный HF кеш
  HF_HUB_OFFLINE  — офлайн режим (только из кеша)

Пример:
    RVC_MODELS_DIR=/data/rvc python xiaozhi_server.py
    HF_TOKEN=hf_xxx RVC_MODELS_DIR=/data/rvc python xiaozhi_server.py

CLI (после pip install rvc-py):
    rvc-download --models rmvpe hubert
    rvc-download --models rmvpe --dir /data/rvc
"""
from __future__ import annotations

import os
import shutil

# Таблица: имя файла → (hf_repo_id, filename_in_repo)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "hubert_base.pt": (
        "lj1995/VoiceConversionWebUI",
        "hubert_base.pt",
    ),
    "rmvpe.pt": (
        "lj1995/VoiceConversionWebUI",
        "rmvpe.pt",
    ),
    "rmvpe.onnx": (
        "lj1995/VoiceConversionWebUI",
        "rmvpe.onnx",
    ),
}


def get_models_dir() -> str:
    """
    Возвращает директорию для хранения весовых файлов RVC.

    Приоритет:
      1. RVC_MODELS_DIR из окружения
      2. ~/.cache/rvc/models
    """
    env = os.environ.get("RVC_MODELS_DIR")
    path = os.path.expanduser(env) if env else os.path.join(
        os.path.expanduser("~"), ".cache", "rvc", "models"
    )
    os.makedirs(path, exist_ok=True)
    return path


def download_model(model_name: str, out_dir: str | None = None) -> str:
    """
    Скачать модель через huggingface_hub.

    Файл скачивается в HF кеш (~/.cache/huggingface/hub),
    затем копируется в out_dir (или RVC_MODELS_DIR).

    Args:
        model_name: Ключ из MODEL_REGISTRY (например 'hubert_base.pt').
        out_dir: Куда положить файл. None = get_models_dir().

    Returns:
        Абсолютный путь к файлу в out_dir.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"[RVC] Unknown model: '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    target_dir = out_dir or get_models_dir()
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, model_name)

    if os.path.exists(target_path):
        return target_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub не установлен: pip install huggingface-hub"
        ) from e

    repo_id, filename = MODEL_REGISTRY[model_name]
    print(f"[RVC] Downloading {model_name} from {repo_id} ...")

    # hf_hub_download кладёт файл в HF кеш и возвращает путь
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        # token подхватывается из HF_TOKEN или huggingface-cli login
    )

    # Копируем/симлинкуем из HF кеша в RVC_MODELS_DIR
    try:
        os.symlink(cached_path, target_path)
    except (OSError, NotImplementedError):
        # Windows или нет прав на symlink — просто копируем
        shutil.copy2(cached_path, target_path)

    print(f"[RVC] Ready: {target_path}")
    return target_path


def download_all_models(
    models: list[str] | None = None,
    out_dir: str | None = None,
) -> dict[str, str]:
    """
    Скачать несколько моделей.

    Args:
        models: Список ключей из MODEL_REGISTRY. None = все.
        out_dir: Куда положить. None = get_models_dir().

    Returns:
        dict {model_name: path}
    """
    names = models or list(MODEL_REGISTRY)
    target = out_dir or get_models_dir()
    print(f"[RVC] Models dir: {target}")
    results = {}
    for name in names:
        results[name] = download_model(name, out_dir=target)
    return results


# Алиас для rvc-download CLI (см. _cli.py)
download = download_all_models


if __name__ == "__main__":
    download_all_models()
