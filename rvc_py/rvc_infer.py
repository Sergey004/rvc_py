from __future__ import annotations

import os
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from .download_models import download_model
from .f0_extractor import extract_f0
from .hubert_contentvec import Hubert
from .rmvpe_extractor import extract_f0_rmvpe
from .rvc_model import RVCModel

_rvc_cache: dict[tuple, tuple[Hubert, RVCModel]] = {}


def _resolve_device(device: str) -> str:
    """Normalize device string; fall back to CPU if CUDA is unavailable."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[RVC] CUDA unavailable, falling back to CPU (requested: {device})")
        return "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


def _ensure_models_dir() -> str:
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models", "RVC")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _get_or_create_models(
    rvc_model_path: str,
    device: str,
    hubert_path: Optional[str],
    index_path: Optional[str],
    fp16: bool,
    index_rate: float,
) -> tuple[Hubert, RVCModel]:
    """Return cached (hubert, rvc) pair, loading from disk if needed."""
    cache_key = (rvc_model_path, index_path, fp16, device)
    if cache_key in _rvc_cache:
        print(f"[RVC] Using cached models on {device}")
        return _rvc_cache[cache_key]

    models_dir = _ensure_models_dir()

    if hubert_path is None:
        hubert_path = os.path.join(models_dir, "hubert_base.pt")
    if not os.path.exists(hubert_path):
        print("[RVC] HuBERT not found, downloading...")
        hubert_path = download_model("hubert_base.pt", out_dir=models_dir)

    print(f"[RVC] Loading HuBERT: {hubert_path}")
    hubert = Hubert(hubert_path, device=device)

    print(f"[RVC] Loading RVC model: {rvc_model_path}")
    rvc = RVCModel(rvc_model_path, device=device, index_path=index_path, fp16=fp16)
    if index_path and index_rate > 0.0:
        rvc.set_index(index_path, index_rate=index_rate)

    _rvc_cache[cache_key] = (hubert, rvc)
    print(f"[RVC] Initialized on {device}")
    return hubert, rvc


def _extract_features(
    hubert: Hubert,
    wav: np.ndarray,
    sr: int,
    device: str,
    version: str,
) -> torch.Tensor:
    """Extract HuBERT features, returning (1, T, C) tensor."""
    wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    if wav16.ndim == 1:
        wav16 = wav16[None, :]
    elif wav16.ndim > 2:
        wav16 = wav16.reshape(1, -1)
    wav16_tensor = torch.from_numpy(wav16).to(device)

    padding_mask = torch.zeros(wav16_tensor.shape, dtype=torch.bool, device=device)
    output_layer = 9 if version == "v1" else 12
    with torch.no_grad():
        logits = hubert.model.extract_features(
            source=wav16_tensor,
            padding_mask=padding_mask,
            output_layer=output_layer,
        )
        units = hubert.model.final_proj(logits[0]) if version == "v1" else logits[0]
    return F.interpolate(units.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)


def _extract_f0(
    wav: np.ndarray,
    sr: int,
    f0_method: str,
    device: str,
    rmvpe_model_path: Optional[str],
) -> np.ndarray:
    """Extract fundamental frequency in Hz."""
    if f0_method == "rmvpe":
        models_dir = _ensure_models_dir()
        if rmvpe_model_path is None:
            rmvpe_model_path = os.path.join(models_dir, "rmvpe.pt")
        if not os.path.exists(rmvpe_model_path):
            print("[RVC] RMVPE not found, downloading...")
            rmvpe_model_path = download_model("rmvpe.pt", out_dir=models_dir)
        return extract_f0_rmvpe(wav, sr, rmvpe_model_path, device=device)
    return extract_f0(wav, sr, method=f0_method, device=device)


def _f0_to_coarse(f0_hz: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert Hz array to coarse pitch (1..255) and aligned fine pitch."""
    src_len = f0_hz.shape[0]
    if src_len != target_len:
        x = np.arange(src_len, dtype=np.float32)
        xp = np.linspace(0, src_len - 1, num=target_len, dtype=np.float32)
        f0_rs = np.interp(xp, x, f0_hz).astype(np.float32)
    else:
        f0_rs = f0_hz.astype(np.float32)

    f0_min, f0_max = 50, 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + np.clip(f0_rs, 0, None) / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel = np.clip(f0_mel, 1, 255)
    f0_coarse = np.rint(f0_mel).astype(np.int32)

    return f0_coarse, f0_rs


def rvc_infer(
    wav: np.ndarray,
    sr: int,
    rvc_model_path: str,
    device: str = "cuda",
    hubert_path: Optional[str] = None,
    f0_method: str = "rmvpe",
    rmvpe_model_path: Optional[str] = None,
    index_path: Optional[str] = None,
    index_rate: float = 0.0,
    fp16: bool = False,
    pitch_shift: int = 0,
    use_index: bool = False,
    sample_rate: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """Run RVC voice conversion.

    Args:
        wav: Input audio samples (float32, mono or stereo).
        sr: Input sample rate.
        rvc_model_path: Path to the RVC .pth model file.
        device: Target device ("cuda", "cuda:0", "cpu").
        hubert_path: Path to HuBERT checkpoint (auto-downloaded if None).
        f0_method: F0 extraction method ("rmvpe" or "torchcrepe").
        rmvpe_model_path: Path to RMVPE checkpoint (auto-downloaded if None).
        index_path: Path to Faiss index for retrieval blending.
        index_rate: Blending rate for retrieval (0.0 = disabled).
        fp16: Use half-precision inference.
        pitch_shift: Pitch shift in semitones.
        use_index: Enable Faiss retrieval blending.
        sample_rate: Override output sample rate (None = use model default).

    Returns:
        (output_audio, output_sample_rate) tuple.
    """
    device = _resolve_device(device)
    hubert, rvc = _get_or_create_models(
        rvc_model_path, device, hubert_path, index_path, fp16, index_rate
    )

    # Extract HuBERT features
    units = _extract_features(hubert, wav, sr, device, rvc.version)

    # Extract F0
    f0_hz = _extract_f0(wav, sr, f0_method, device, rmvpe_model_path)
    if pitch_shift != 0:
        f0_hz = f0_hz * (2 ** (pitch_shift / 12))

    # Build pitch tensors
    f0_coarse, f0_rs = _f0_to_coarse(f0_hz, units.shape[1])
    pitch = torch.tensor(f0_coarse, device=device).unsqueeze(0).long()
    pitchf = torch.tensor(f0_rs, device=device).unsqueeze(0).float()

    # Run model inference
    out = rvc.infer(units, pitch=pitch, pitchf=pitchf, sid=0, use_index=use_index)
    wav_out = out[0] if isinstance(out, tuple) else out
    wav_out = wav_out.detach().cpu().numpy().squeeze()

    out_sr = rvc.sample_rate if sample_rate is None else sample_rate
    return wav_out, out_sr
