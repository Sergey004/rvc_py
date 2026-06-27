from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import faiss
except ImportError:
    faiss = None
    print("[RVC] faiss not installed — retrieval blending unavailable.")

from .lib.infer_pack.models_dml import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)


class RVCModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        index_path: Optional[str] = None,
        fp16: bool = False,
        sample_rate: Optional[int] = None,
    ):
        super().__init__()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        cpt = checkpoint
        version = cpt.get("version", "v1")
        if_f0 = cpt.get("f0", 1)
        config = cpt["config"]
        self.sample_rate = sample_rate or config[-1]
        is_half = fp16

        if version == "v1":
            if if_f0 == 1:
                self.model = SynthesizerTrnMs256NSFsid(*config, is_half=is_half)
            else:
                self.model = SynthesizerTrnMs256NSFsid_nono(*config)
        elif version == "v2":
            if if_f0 == 1:
                self.model = SynthesizerTrnMs768NSFsid(*config, is_half=is_half)
            else:
                self.model = SynthesizerTrnMs768NSFsid_nono(*config)
        else:
            raise ValueError(f"Unknown model version: {version}")

        weight = cpt.get("weight", cpt)
        self.model.load_state_dict(weight, strict=False)
        self.model.eval()
        self.model = self.model.to(device)
        if is_half:
            self.model = self.model.half()

        self.device = device
        self.fp16 = fp16
        self.if_f0 = if_f0
        self.version = version

        self.index = None
        self.big_npy = None
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            try:
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            except Exception:
                self.big_npy = None
        self.index_rate = 0.0

    def set_index(self, index_path: str, index_rate: float = 0.5) -> None:
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            try:
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            except Exception:
                self.big_npy = None
            self.index_rate = index_rate

    def infer(
        self,
        units: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        sid: int = 0,
        use_index: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Run voice conversion inference.

        Args:
            units: HuBERT features (1, T, C).
            pitch: Coarse pitch (1, T) as long tensor.
            pitchf: Fine pitch in Hz (1, T) as float tensor.
            sid: Speaker ID.
            use_index: Whether to apply Faiss retrieval blending.

        Returns:
            Model output tuple (wav, ...).
        """
        with torch.no_grad():
            if self.fp16:
                units = units.half()
                if pitchf is not None:
                    pitchf = pitchf.half()

            # Faiss retrieval blending
            if use_index and self.index is not None and self.index_rate > 0.0:
                units_np = units.detach().cpu().numpy().squeeze(0)
                npy = units_np.astype("float32") if units_np.dtype != np.float32 else units_np
                try:
                    k = min(8, self.index.ntotal)
                    score, ix = self.index.search(npy, k=k)
                    if self.big_npy is None:
                        self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                    gather = self.big_npy[ix]
                    score = np.maximum(score, 1e-6)
                    weight = 1.0 / (score**2)
                    weight /= weight.sum(axis=1, keepdims=True)
                    blend = np.sum(gather * weight[..., None], axis=1)
                    if self.fp16:
                        blend = blend.astype("float16")
                    units = torch.from_numpy(
                        self.index_rate * blend + (1 - self.index_rate) * units_np
                    ).unsqueeze(0).to(self.device)
                    units = units.to(torch.float16) if self.fp16 else units.to(torch.float32)
                except Exception:
                    pass

            phone_lengths = torch.tensor([units.shape[1]], device=self.device).long()
            sid_tensor = torch.tensor([sid], device=self.device).long()

            if self.if_f0 == 1 and pitch is not None and pitchf is not None:
                return self.model.infer(units, phone_lengths, pitch, pitchf, sid_tensor)
            return self.model.infer(units, phone_lengths, sid_tensor)
