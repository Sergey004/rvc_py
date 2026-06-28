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

_ENCODER_BACKEND = os.environ.get("RVC_ENCODER", "hubert").lower().strip()


class _ContentVecCompat(nn.Module):
    """
    Враппер над transformers HubertModel, который эмулирует fairseq API:
      - extract_features(source, padding_mask, output_layer) → (tensor,)
      - final_proj: nn.Linear(768, 256) если есть в чекпоинте

    Это нужно чтобы rvc_infer._extract_features не менялся — он ожидает
    fairseq-совместимый интерфейс.
    """

    def __init__(self, hf_model: nn.Module, ckpt_path_or_id: str):
        super().__init__()
        self.hf_model = hf_model

        # Пробуем загрузить final_proj из чекпоинта (есть в lengyue233/content-vec-best)
        self.final_proj = self._load_final_proj(ckpt_path_or_id)

    def _load_final_proj(self, source: str) -> nn.Linear | None:
        """
        Пытается вытащить final_proj.weight / final_proj.bias из чекпоинта.
        Возвращает nn.Linear или None если не нашёл.
        """
        try:
            # HuggingFace кеш — ищем pytorch_model.bin или model.safetensors
            import os
            state: dict | None = None

            if os.path.isfile(source):
                # Локальный .bin
                state = torch.load(source, map_location="cpu", weights_only=True)
            else:
                # HuggingFace hub id — грузим через snapshot
                from huggingface_hub import hf_hub_download
                try:
                    path = hf_hub_download(source, "pytorch_model.bin")
                    state = torch.load(path, map_location="cpu", weights_only=True)
                except Exception:
                    try:
                        from safetensors.torch import load_file
                        path = hf_hub_download(source, "model.safetensors")
                        state = load_file(path, device="cpu")
                    except Exception:
                        pass

            if state is None:
                return None

            w_key = "final_proj.weight"
            b_key = "final_proj.bias"
            if w_key in state:
                out_dim, in_dim = state[w_key].shape
                proj = nn.Linear(in_dim, out_dim, bias=b_key in state)
                proj.weight = nn.Parameter(state[w_key])
                if b_key in state:
                    proj.bias = nn.Parameter(state[b_key])
                print(f"[RVC] ContentVec: final_proj loaded ({in_dim}→{out_dim})")
                return proj

        except Exception as e:
            print(f"[RVC] ContentVec: final_proj not loaded ({e}), using hidden state directly")

        return None

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
        output_layer: int,
    ) -> tuple[torch.Tensor]:
        """
        Эмулирует fairseq extract_features API.
        Возвращает кортеж (hidden_state,) — как fairseq логиты[0].
        """
        attention_mask = (~padding_mask).long()
        out = self.hf_model(
            input_values=source,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # hidden_states[0] = embedding, [1..13] = transformer layers
        # output_layer=9 (v1) или 12 (v2) — берём соответствующий слой
        n = len(out.hidden_states)
        idx = min(output_layer, n - 1)
        hidden = out.hidden_states[idx]  # (1, T', 768)

        return (hidden,)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Прямой вызов без padding_mask (для совместимости с Hubert.forward)."""
        padding_mask = torch.zeros(source.shape, dtype=torch.bool, device=source.device)
        return self.extract_features(source, padding_mask, output_layer=12)[0]


class Hubert(nn.Module):
    """
    Универсальный враппер feature extractor для RVC.

    RVC_ENCODER=hubert      → fairseq HuBERT (ckpt_path обязателен)
    RVC_ENCODER=contentvec  → HuggingFace ContentVec (ckpt_path игнорируется)
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

    def _init_hubert(self, ckpt_path: str):
        try:
            import fairseq
            from fairseq.data.dictionary import Dictionary
            import torch.serialization
            torch.serialization.add_safe_globals([Dictionary])
        except ImportError as e:
            raise ImportError(
                "fairseq не установлен. Установите: pip install fairseq\n"
                "Или переключитесь: RVC_ENCODER=contentvec"
            ) from e

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path], suffix=""
        )
        self.model = models[0]
        print(f"[RVC] Encoder: HuBERT (fairseq) ← {ckpt_path}")

    def _init_contentvec(self):
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers не установлен: pip install transformers"
            ) from e

        model_id = os.environ.get("RVC_CONTENTVEC_MODEL", "lengyue233/content-vec-best")
        print(f"[RVC] Encoder: ContentVec (transformers) ← {model_id}")

        hf_model = AutoModel.from_pretrained(model_id)
        # Оборачиваем в fairseq-совместимый враппер
        self.model = _ContentVecCompat(hf_model, model_id)

    def forward(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features из 16kHz аудио тензора (1, T) → (1, T', C)."""
        with torch.no_grad():
            return self.model(wav_tensor)
