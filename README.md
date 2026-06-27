# rvc_py

Minimalist RVC (Retrieval-based Voice Conversion) inference package for Python, adapted from [rvc-python](https://github.com/daswer123/rvc-python).

## Features

- Sample rate support: 32kHz, 40kHz, 48kHz (determined by model).
- F0 extraction: rmvpe, torchcrepe.
- Pitch shift in semitones.
- Faiss retrieval blending for improved voice quality.
- FP16 half-precision inference.
- Automatic model downloading from HuggingFace.

## Installation

```shell
pip install -r requirements.txt
```

## Usage

```python
from rvc_py import rvc_infer

wav_out, sr = rvc_infer(
    wav=audio_array,
    sr=sample_rate,
    rvc_model_path="path/to/model.pth",
    device="cuda",
    f0_method="rmvpe",
)
```

## Architecture

- `rvc_infer.py` — main inference pipeline.
- `rvc_model.py` — RVC model wrapper with Faiss retrieval.
- `hubert_contentvec.py` — ContentVec HuBERT feature extractor.
- `f0_extractor.py` — F0 extraction (torchcrepe).
- `rmvpe_extractor.py` — RMVPE F0 extractor (neural network based).
- `download_models.py` — automatic model downloader.
- `lib/infer_pack/` — synthesizer model architectures.

## References

- [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [ContentVec](https://github.com/auspicious3000/contentvec)
