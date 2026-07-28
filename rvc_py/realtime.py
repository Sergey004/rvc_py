"""
rvc_py/realtime.py — Streaming RVC pipeline для интеграции с TTS.

Использование:
    from rvc_py.realtime import RVCStreamer

    streamer = RVCStreamer(
        rvc_model_path="freddy.pth",
        device="cuda",
        f0_up_key=-5,
        input_sr=24000,   # Chatterbox SR
        output_sr=16000,  # XiaoZhi SR
    )

    # В цикле стриминга:
    converted = streamer.push(chunk_float32)  # None пока буфер не накопился
    tail = streamer.flush()                    # Остаток в конце фразы
    streamer.reset()                           # Между фразами
"""
from __future__ import annotations

import logging
from math import gcd

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger("rvc_streamer")


class RVCStreamer:
    """
    Push-based streaming RVC конвертор.

    Принимает произвольные float32 numpy чанки (любой длины),
    аккумулирует в буфер, выдаёт конвертированные чанки когда
    накоплено достаточно данных для одного RVC inference блока.

    Не требует sounddevice — работает как чистый audio transform.
    Потокобезопасен для использования из asyncio (inference синхронный,
    вызывать через run_in_executor).
    """

    def __init__(
        self,
        rvc_model_path: str,
        device: str = "cuda",
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_path: str | None = None,
        index_rate: float = 0.0,
        fp16: bool = False,
        block_size: int = 16000,   # сэмплов @ 16kHz = 1 секунда
        overlap: int = 2048,        # контекст между блоками
        crossfade: int = 512,       # зона плавного перехода
        input_sr: int = 24000,      # SR входного аудио (Chatterbox = 24kHz)
        output_sr: int = 16000,     # SR выходного аудио (XiaoZhi = 16kHz)
    ):
        self.rvc_model_path = rvc_model_path
        self.device = device
        self.f0_up_key = f0_up_key
        self.f0_method = f0_method
        self.index_path = index_path
        self.index_rate = index_rate
        self.fp16 = fp16
        self.block_size = block_size
        self.overlap = overlap
        self.input_sr = input_sr
        self.output_sr = output_sr

        # Кроссфейд
        self._crossfade = crossfade
        self._prev_tail = np.zeros(crossfade, dtype=np.float32)
        self._fade_in = np.linspace(0, 1, crossfade, dtype=np.float32)
        self._fade_out = np.linspace(1, 0, crossfade, dtype=np.float32)

        # Буфер накапливает данные @ 16kHz
        self._buf = np.zeros(0, dtype=np.float32)

        # Реальный выходной SR самой RVC-модели (обычно 40000/48000, зависит
        # от архитектуры) — узнаётся только после загрузки модели в _ensure_models().
        # Дефолт 16000 ниже используется только в fallback-ветке _infer_block,
        # если модель так и не загрузилась ни разу (без этого падёт с AttributeError).
        self._native_sr = 16000

        # Модели — ленивая инициализация через кеш rvc_infer
        self._models_ready = False

    # ── Инициализация моделей ─────────────────────────────────────────────────

    def _ensure_models(self):
        """Ленивая загрузка через кеш _get_or_create_models из rvc_infer."""
        if self._models_ready:
            return
        from rvc_py.rvc_infer import _get_or_create_models
        self._hubert, self._rvc = _get_or_create_models(
            self.rvc_model_path,
            self.device,
            hubert_path=None,       # auto-download
            index_path=self.index_path,
            fp16=self.fp16,
            index_rate=self.index_rate,
        )
        self._models_ready = True
        self._native_sr = self._rvc.sample_rate
        logger.info(
            f"RVCStreamer ready: {self.rvc_model_path} on {self.device} "
            f"(native output SR: {self._native_sr})"
        )

    # ── Ресэмплинг ───────────────────────────────────────────────────────────

    def _resample(self, audio: np.ndarray, src: int, dst: int) -> np.ndarray:
        if src == dst:
            return audio
        g = gcd(src, dst)
        return resample_poly(audio, dst // g, src // g).astype(np.float32)

    # ── Кроссфейд на стыках блоков ───────────────────────────────────────────

    def _apply_crossfade(self, chunk: np.ndarray) -> np.ndarray:
        cf = self._crossfade
        out = chunk.copy()
        if len(out) >= cf and np.any(self._prev_tail != 0):
            out[:cf] = chunk[:cf] * self._fade_in + self._prev_tail * self._fade_out
        self._prev_tail = chunk[-cf:].copy() if len(chunk) >= cf else np.zeros(cf, dtype=np.float32)
        return out

    # ── RVC inference одного блока ────────────────────────────────────────────

    def _infer_block(self, frame_16k: np.ndarray) -> np.ndarray:
        """Запускает RVC inference на одном блоке @ 16kHz.
        ВНИМАНИЕ: возвращает аудио НЕ @ 16kHz, а @ self._native_sr —
        реальном выходном SR самой RVC-модели (обычно 40000/48000).
        Ответственность за финальный ресэмпл — на вызывающем push()/flush().
        """
        import torch
        from rvc_py.rvc_infer import _extract_features, _extract_f0, _f0_to_coarse

        try:
            self._ensure_models()

            units = _extract_features(
                self._hubert, frame_16k, 16000,
                self.device, self._rvc.version,
            )

            f0_hz = _extract_f0(
                frame_16k, 16000,
                self.f0_method, self.device,
                rmvpe_model_path=None,  # auto-download / кеш
            )
            if self.f0_up_key != 0:
                f0_hz = f0_hz * (2.0 ** (self.f0_up_key / 12.0))

            f0_coarse, f0_rs = _f0_to_coarse(f0_hz, units.shape[1])
            pitch  = torch.tensor(f0_coarse, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(f0_rs,     device=self.device).unsqueeze(0).float()

            use_idx = self.index_path is not None and self.index_rate > 0.0
            out = self._rvc.infer(units, pitch=pitch, pitchf=pitchf, sid=0, use_index=use_idx)
            wav = out[0] if isinstance(out, tuple) else out
            return wav.detach().cpu().numpy().squeeze().astype(np.float32)

        except Exception as e:
            logger.warning(f"RVC infer error, passthrough: {e}")
            # Fallback: пропустить без конвертации
            return frame_16k[self.overlap:] if len(frame_16k) > self.overlap else frame_16k

    # ── Публичный API ─────────────────────────────────────────────────────────

    def push(self, audio_chunk: np.ndarray) -> np.ndarray | None:
        """
        Принимает чанк float32 в input_sr.
        Возвращает конвертированный float32 в output_sr, или None
        если буфер ещё не накопил достаточно данных.
        """
        # Ресэмпл в 16kHz для RVC
        chunk_16k = self._resample(audio_chunk, self.input_sr, 16000)
        self._buf = np.concatenate([self._buf, chunk_16k])

        needed = self.block_size + self.overlap
        if len(self._buf) < needed:
            return None

        # Берём block_size + overlap для контекста
        frame = self._buf[:needed]
        # Сдвигаем буфер, сохраняем overlap как контекст
        self._buf = self._buf[self.block_size:]

        # Inference
        wav = self._infer_block(frame)

        # Trim overlap с начала выхода (пропорционально)
        trim = int(self.overlap * len(wav) / needed)
        wav = wav[trim:]

        # Кроссфейд
        wav = self._apply_crossfade(wav)

        # Ресэмпл из реального SR модели (НЕ 16000!) в output_sr
        return self._resample(wav, self._native_sr, self.output_sr)

    def flush(self) -> np.ndarray | None:
        """
        Обработать остаток буфера после окончания TTS фразы.
        Вызывать после последнего push().
        """
        if len(self._buf) < 512:
            self._buf = np.zeros(0, dtype=np.float32)
            return None

        frame = self._buf.copy()
        self._buf = np.zeros(0, dtype=np.float32)

        wav = self._infer_block(frame)
        wav = self._apply_crossfade(wav)
        return self._resample(wav, self._native_sr, self.output_sr)

    def reset(self):
        """Сбросить состояние между фразами."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._prev_tail = np.zeros(self._crossfade, dtype=np.float32)
