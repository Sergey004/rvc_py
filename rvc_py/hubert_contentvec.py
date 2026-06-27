from __future__ import annotations

import fairseq
import torch
import torch.nn as nn
from fairseq.data.dictionary import Dictionary
import torch.serialization

torch.serialization.add_safe_globals([Dictionary])


class Hubert(nn.Module):
    """ContentVec HuBERT wrapper for RVC inference."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        super().__init__()
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path], suffix=""
        )
        self.model = models[0]
        self.model.eval()
        self.device = device
        self.model.to(device)

    def forward(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features from 16kHz audio tensor.

        Args:
            wav_tensor: (1, T) float32 tensor at 16kHz.

        Returns:
            Feature tensor.
        """
        with torch.no_grad():
            return self.model.extract_features(wav_tensor)[0]
