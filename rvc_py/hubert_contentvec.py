"""
hubert_contentvec.py — HuBERT / ContentVec feature extractor для RVC.

Переключение через переменную окружения:

    RVC_ENCODER=hubert       # fairseq HuBERT (по умолчанию, оригинальный RVC)
    RVC_ENCODER=contentvec   # HuggingFace ContentVec (быстрее, без fairseq)

Пример:
    RVC_ENCODER=contentvec python xiaozhi_server.py
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn

# Читаем свич один раз при импорте модуля
_ENCODER_BACKEND = os.environ.get("RVC_ENCODER", "hubert").lower().strip()


class Hubert(nn.Module):
    """
    Универсальный враппер feature extractor для RVC.
    Поддерживает два бэкенда через RVC_ENCODER:
      - 'hubert'      : fairseq HuBERT (ckpt_path обязателен)
      - 'contentvec'  : HuggingFace lengyue233/content-vec-best
                        (ckpt_path игнорируется, скачивается автоматически)
    """

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._backend = _ENCODER_BACKEND

        if self._backend == "contentvec":
            self._init_contentvec()
        else:
            self._init_hubert(ckpt_path)

        self.model.eval()
        self.model.to(device)

    # ── Инициализация бэкендов ────────────────────────────────────────────

    def _init_hubert(self, ckpt_path: str):
        """Оригинальный fairseq HuBERT — требует fairseq."""
        try:
            import fairseq
            from fairseq.data.dictionary import Dictionary
            import torch.serialization
            torch.serialization.add_safe_globals([Dictionary])
        except ImportError as e:
            raise ImportError(
                "fairseq не установлен. Установите: pip install fairseq\n"
                "Или переключитесь на ContentVec: RVC_ENCODER=contentvec"
            ) from e

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path], suffix=""
        )
        self.model = models[0]
        print(f"[RVC] Encoder: HuBERT (fairseq) ← {ckpt_path}")

    def _init_contentvec(self):
        """ContentVec через HuggingFace transformers — без fairseq."""
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers не установлен. Установите: pip install transformers\n"
                "Или переключитесь обратно: RVC_ENCODER=hubert"
            ) from e

        model_id = os.environ.get("RVC_CONTENTVEC_MODEL", "lengyue233/content-vec-best")
        print(f"[RVC] Encoder: ContentVec (transformers) ← {model_id}")
        self.model = AutoModel.from_pretrained(model_id)

    # ── Forward — единый интерфейс для обоих бэкендов ────────────────────

    def forward(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        """
        Извлечь признаки из 16kHz аудио тензора.

        Args:
            wav_tensor: (1, T) float32 @ 16kHz

        Returns:
            Feature tensor (1, T', C)
        """
        with torch.no_grad():
            if self._backend == "contentvec":
                return self._forward_contentvec(wav_tensor)
            return self._forward_hubert(wav_tensor)

    def _forward_hubert(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        return self.model.extract_features(wav_tensor)[0]

    def _forward_contentvec(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        # ContentVec через HuggingFace: нужен padding_mask
        padding_mask = torch.zeros(wav_tensor.shape, dtype=torch.bool, device=self.device)
        out = self.model(
            input_values=wav_tensor,
            attention_mask=(~padding_mask).long(),
        )
        # last_hidden_state: (1, T', C)
        return out.last_hidden_state
