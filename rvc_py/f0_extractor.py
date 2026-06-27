from __future__ import annotations

import numpy as np
import librosa


def extract_f0(wav: np.ndarray, sr: int, method: str = "rmvpe", device: str = "cpu") -> np.ndarray:
    """Extract fundamental frequency from audio.

    Args:
        wav: Audio samples (float32, mono).
        sr: Sample rate.
        method: F0 extraction method ("torchcrepe").
        device: Device for computation.

    Returns:
        F0 array in Hz, shape (T,).
    """
    if method == "torchcrepe":
        import torchcrepe

        wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
        _, f0, _, _ = torchcrepe.predict(wav16, 16000, viterbi=True, step_size=10)
        return f0
    raise NotImplementedError(f"F0 extraction method '{method}' is not supported")
