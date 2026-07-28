"""
rvc_py/realtime.py — Streaming RVC pipeline для интеграции с TTS и realtime-аудио.

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

Алгоритмы переняты из RVC-realtime-voice-changer (gui_v1.py / rtrvc.py):
  - SOLA (Shape-Optimized Lapped Algorithm) — оптимальный сдвиг crossfade
    вместо линейного, устраняет фазовые артефакты/эхо на стыках блоков.
  - Pitch cache — плавный pitch-контур между блоками.
  - Formant shift — сохранение тембра при pitch-shift.
  - RMS mix rate — подмес огибающей громкости оригинала.
  - VAD threshold — gate тишины перед inference (экономит GPU).
  - Extra frame — доп. контекст слева для стабильности HuBERT/F0.
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
        crossfade: int = 512,       # зона плавного перехода (legacy linear crossfade)
        input_sr: int = 24000,      # SR входного аудио (Chatterbox = 24kHz)
        output_sr: int = 16000,     # SR выходного аудио (XiaoZhi = 16kHz)
        # ── Новые параметры (перенято из rtrvc.py / gui_v1.py) ──────────────
        vad_threshold_db: float | None = -45.0,
        # SOLA (Shape-Optimized Lapped Algorithm) — поиск оптимального offset
        # в зоне crossfade через argmax(cor_nom/cor_den). Устраняет фазовые
        # артефакты. use_sola=False — обратно совместимый линейный crossfade.
        use_sola: bool = True,
        sola_search_frame: int = 256,  # зона поиска SOLA @ 16kHz (сэмплов)
        # Formant shift в полутонах (0 = выключено). Сохраняет тембр диктора
        # при pitch-shift через компенсацию return_length2 scaling.
        formant_shift: float = 0.0,
        # RMS mix rate: 1.0 = чистый выход модели (дефолт), <1.0 = больше
        # оригинальной огибающей громкости (RVC-выход часто "сжат" по динамике).
        rms_mix_rate: float = 1.0,
        # Extra frame — доп. контекст слева от блока (сэмплов @ 16kHz).
        # 0 = выключено. Уменьшает краевые артефакты HuBERT/F0 на стыках.
        # Рекомендуется 40000 (2.5s @ 16kHz) для длинных фраз.
        extra_frame: int = 0,
        # Защита глухих/свистящих от заглушения. 0.5 = выключено.
        protect: float = 0.5,
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

        # ── VAD ─────────────────────────────────────────────────────────────
        # None = нет гейта; иначе блоки с RMS ниже порога (в dB FS) замещаются
        # тишиной (намного дешевле inference на GPU, не "плывёт" pitch).
        self.vad_threshold_db = vad_threshold_db

        # ── SOLA ────────────────────────────────────────────────────────────
        self.use_sola = use_sola
        self.sola_search_frame = sola_search_frame
        # sola_buffer — зона кроссфейда, по размеру = crossfade (если use_sola=True)
        self._crossfade = crossfade
        self._sola_buffer = np.zeros(crossfade, dtype=np.float32)
        # Окна для SOLA fallback к линейному (если use_sola=False)
        self._fade_in = np.linspace(0, 1, crossfade, dtype=np.float32)
        self._fade_out = np.linspace(1, 0, crossfade, dtype=np.float32)

        # ── Pitch cache ─────────────────────────────────────────────────────
        # Сохраняем pitch-контур между блоками (как rtrvc.py:89-94,414-417),
        # избегая "скачков" pitch на стыках. Кеш — кольцевой, на 1024 кадров.
        self._cache_pitch = np.zeros(1024, dtype=np.int64)
        self._cache_pitchf = np.zeros(1024, dtype=np.float32)

        # ── Formant / RMS / protect ─────────────────────────────────────────
        self.formant_shift = formant_shift
        self.rms_mix_rate = rms_mix_rate
        self.protect = protect

        # ── Extra frame ─────────────────────────────────────────────────────
        # Доп. контекст слева от текущего блока для стабильности HuBERT/F0.
        self.extra_frame = extra_frame
        self._extra_buffer = np.zeros(extra_frame, dtype=np.float32) if extra_frame > 0 else None

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

    # ── SOLA crossfade (Shape-Optimized Lapped Algorithm) ─────────────────────
    #
    # Перенято из gui_v1.py:979-1012. Идея: вместо линейного crossfade с фиксированным
    # offset=0, ищем такой сдвиг в окне [0, sola_search_frame], который максимизирует
    # нормированную корреляцию между "хвостом" предыдущего блока (sola_buffer) и
    # началом нового. Это устраняет фазовые артефакты/эхо на стыках.
    #
    # Реализация через numpy — без torch-зависимости в fast-path (torch тут не нужен).

    def _apply_sola(self, chunk: np.ndarray) -> np.ndarray:
        """SOLA crossfade: поиск оптимального offset, затем оконный overlap-add.

        Перенято из gui_v1.py:979-1012. В отличие от rtrvc, у нас chunk — это
        уже "новый" выход блока (после trim overlap). ищем сдвиг в [0, search],
        который максимизирует норм-ю корреляцию chunk[k:k+cf] с sola_buffer.
        Затем crossfade по年的时间 и сохраняем "хвост" chunk'а для следующего вызова.
        """
        cf = self._crossfade
        if len(chunk) < cf or not np.any(self._sola_buffer):
            # Первый блок или размер слишком мал — fallback к линейному crossfade
            out = chunk.copy()
            if len(out) >= cf:
                out[:cf] = chunk[:cf] * self._fade_in + self._sola_buffer * self._fade_out
            self._sola_buffer = chunk[-cf:].copy() if len(chunk) >= cf else np.zeros(cf, dtype=np.float32)
            return out

        # Зона поиска: [0, sola_search_frame], но не длиннее чем позволяет chunk
        search = min(self.sola_search_frame, len(chunk) - cf)
        if search <= 0:
            out = chunk.copy()
            out[:cf] = chunk[:cf] * self._fade_in + self._sola_buffer * self._fade_out
            self._sola_buffer = chunk[-cf:].copy()
            return out

        # Нормированная корреляция для каждого кандидата offset в [0..search]:
        #   cor_nom[k] = sum(chunk[k:k+cf] * sola_buffer)
        #   cor_den[k] = sqrt(sum(chunk[k:k+cf]^2) + 1e-8)
        # argmax(cor_nom/cor_den) — оптимальный offset.
        # Простая numpy-реализация (намного быстрее FFT на малых размерах).
        sola_buffer = self._sola_buffer.astype(np.float32)
        cor_nom = np.empty(search + 1, dtype=np.float32)
        cor_den = np.empty(search + 1, dtype=np.float32)
        for k in range(search + 1):
            seg = chunk[k : k + cf]
            cor_nom[k] = np.dot(seg, sola_buffer)
            cor_den[k] = np.sqrt(np.dot(seg, seg) + 1e-8)
        ratio = cor_nom / np.maximum(cor_den, 1e-8)
        sola_offset = int(np.argmax(ratio))

        out = chunk[sola_offset:].copy()
        # Crossfade с sola_buffer в начале
        out[:cf] = out[:cf] * self._fade_in + self._sola_buffer * self._fade_out

        # Сохраняем "хвост" чанка как контекст для следующего вызова.
        # Берём последние cf сэмплов — это и есть зона, которая при следующем вызове
        # будет файдиться с началом нового блока.
        if len(out) >= cf:
            self._sola_buffer = out[-cf:].copy()
        else:
            self._sola_buffer = np.zeros(cf, dtype=np.float32)
        return out

    # ── Линейный crossfade (legacy) ────────────────────────────────────────────

    def _apply_crossfade_linear(self, chunk: np.ndarray) -> np.ndarray:
        """Линейный crossfade —_backup когда use_sola=False."""
        cf = self._crossfade
        out = chunk.copy()
        if len(out) >= cf and np.any(self._sola_buffer):
            out[:cf] = chunk[:cf] * self._fade_in + self._sola_buffer * self._fade_out
        self._sola_buffer = chunk[-cf:].copy() if len(chunk) >= cf else np.zeros(cf, dtype=np.float32)
        return out

    def _apply_crossfade(self, chunk: np.ndarray) -> np.ndarray:
        """Диспетчер: SOLA или линейный."""
        if self.use_sola:
            return self._apply_sola(chunk)
        return self._apply_crossfade_linear(chunk)

    # ── VAD gate ──────────────────────────────────────────────────────────────

    def _is_silent(self, audio: np.ndarray) -> bool:
        """RMS-based VAD: True если блок "тихий" (ниже порога в dB FS)."""
        if self.vad_threshold_db is None:
            return False
        if len(audio) == 0:
            return True
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-12
        db = 20.0 * np.log10(rms)
        return db < self.vad_threshold_db

    # ── RMS mix rate — подмес огибающей громкости оригинала ────────────────────

    def _apply_rms_mix(
        self, original_16k: np.ndarray, converted: np.ndarray, original_sr: int, converted_sr: int
    ) -> np.ndarray:
        """Сохранение оригинальной огибающей громкости на RVC-выходе.

        rms_mix_rate=1.0 — чистый выход модели (без вызова).
        rms_mix_rate=0.0 — полностью оригинальная огибающая.
        0<rate<1.0 — интерполяция.
        """
        if self.rms_mix_rate >= 1.0:
            return converted
        try:
            from rvc_py.rvc_infer import _change_rms
            return _change_rms(original_16k, original_sr, converted, converted_sr, self.rms_mix_rate)
        except Exception as e:
            logger.warning(f"RMS mix failed, passthrough: {e}")
            return converted

    # ── RVC inference одного блока ────────────────────────────────────────────

    def _infer_block(self, frame_16k: np.ndarray) -> np.ndarray:
        """Запускает RVC inference на одном блоке @ 16kHz.
        ВНИМАНИЕ: возвращает аудио НЕ @ 16kHz, а @ self._native_sr —
        реальном выходном SR самой RVC-модели (обычно 40000/48000).
        Ответственность за финальный ресэмпл — на вызывающем push()/flush().
        """
        try:
            import torch
            from rvc_py.rvc_infer import _extract_features, _extract_f0, _f0_to_coarse
        except Exception as e:
            logger.warning(f"RVC imports failed, passthrough: {e}")
            return frame_16k[self.overlap:] if len(frame_16k) > self.overlap else frame_16k

        try:
            self._ensure_models()

            # Formant shift: компенсация через return_length2 scaling.
            # factor=2^(formant/12); if_f0 pitch shift уменьшаем на formant_shift
            # (как в rtrvc.py:400-417 — pitch_up_key - formant_shift, return_length2 scaling).
            factor = 2.0 ** (self.formant_shift / 12.0)

            # Extra frame: добавляем контекст слева (резидентный в _extra_buffer)
            # для стабильности HuBERT/F0 на краях блока.
            if self.extra_frame > 0 and self._extra_buffer is not None:
                full_frame = np.concatenate([self._extra_buffer, frame_16k])
                skip_head = len(self._extra_buffer)
            else:
                full_frame = frame_16k
                skip_head = 0
            # Сохраняем "хвост" текущего блока как контекст для следующего.
            if self.extra_frame > 0 and self._extra_buffer is not None:
                needed = self.extra_frame
                if len(frame_16k) >= needed:
                    self._extra_buffer = frame_16k[-needed:].copy()
                else:
                    # Сдвигаем и дополняем
                    shift = len(frame_16k)
                    new_buf = np.zeros(needed, dtype=np.float32)
                    new_buf[:needed - shift] = self._extra_buffer[shift:]
                    new_buf[needed - shift:] = frame_16k
                    self._extra_buffer = new_buf

            units = _extract_features(
                self._hubert, full_frame, 16000,
                self.device, self._rvc.version,
            )

            f0_hz = _extract_f0(
                full_frame, 16000,
                self.f0_method, self.device,
                rmvpe_model_path=None,  # auto-download / кеш
            )
            # Pitch shift с учётом formant-компенсации (как rtrvc.py)
            effective_pitch = self.f0_up_key - self.formant_shift
            if effective_pitch != 0:
                f0_hz = f0_hz * (2.0 ** (effective_pitch / 12.0))

            # Target length с учётом formant scaling (return_length2 в rtrvc.py:401-402)
            p_len_target = units.shape[1]
            return_length2 = int(np.ceil(p_len_target * factor))

            f0_coarse, f0_rs = _f0_to_coarse(f0_hz, p_len_target)

            # ── Pitch cache — плавный контур между блоками ────────────────────
            # Перенято из rtrvc.py:89-94, 413-421 — кольцевой буфер на 1024 кадров
            # (HuBERT frames @ 50Hz). На каждом блоке: сдвигаем кеш влево на
            # shift=p_len (количество новых фреймов в блоке), дописываем новые
            # значения в конец (отбрасывая 3 "граничных" кадра спереди — rtrvc
            # обрезает pitch[3:-1] для стабилизации контура), для inference берём
            # последние p_len кадров.
            p_len = p_len_target
            shift = min(p_len, len(self._cache_pitch))
            cache_len = len(self._cache_pitch)
            if shift < cache_len:
                # Сдвиг кеша влево
                self._cache_pitch[:-shift] = self._cache_pitch[shift:].copy()
                self._cache_pitchf[:-shift] = self._cache_pitchf[shift:].copy()
                # Записываем новые f0 значения в конец, обрезая 3 граничных кадра
                # спереди (как rtrvc.py:416-417, [:4-norm] → [-n:]). Длина записи —
                # ровно shift, чтобы следующие далее чтения последних p_len были
                # консистентны с записанными данными.
                src = f0_coarse[3:] if len(f0_coarse) > shift else f0_coarse
                src_f = f0_rs[3:] if len(f0_rs) > shift else f0_rs
                n_write = min(len(src), shift)
                self._cache_pitch[cache_len - shift : cache_len - shift + n_write] = src[:n_write]
                self._cache_pitchf[cache_len - shift : cache_len - shift + n_write] = src_f[:n_write]
                cache_pitch_np = self._cache_pitch[None, -p_len:]
                # Компенсация formant scaling для pitchf (как rtrvc.py:420)
                cache_pitchf_np = self._cache_pitchf[None, -p_len:] * (return_length2 / p_len)
                pitch = torch.from_numpy(cache_pitch_np).to(self.device).long()
                pitchf = torch.from_numpy(cache_pitchf_np.astype(np.float32)).to(self.device).float()
            else:
                # Блок длиннее кеша — берём как есть (без кеширования)
                pitch = torch.tensor(f0_coarse, device=self.device).unsqueeze(0).long()
                pitchf = (
                    torch.tensor(f0_rs, device=self.device, dtype=torch.float32).unsqueeze(0)
                    * (return_length2 / p_len)
                )

            use_idx = self.index_path is not None and self.index_rate > 0.0
            out = self._rvc.infer(
                units, pitch=pitch, pitchf=pitchf, sid=0,
                use_index=use_idx, protect=self.protect,
            )
            wav = out[0] if isinstance(out, tuple) else out
            wav = wav.detach().cpu().numpy().squeeze().astype(np.float32)

            # Если был extra_frame — отрезаем "skip_head" сэмплы с начала (как rtrvc.py:424-427)
            if self.extra_frame > 0 and skip_head > 0:
                # skip_head @ 16kHz соответствует skip_head * (native_sr/16000) @ native_sr
                skip_samples = int(skip_head * len(wav) / len(full_frame))
                if skip_samples > 0 and len(wav) > skip_samples:
                    wav = wav[skip_samples:]
            return wav

        except Exception as e:
            logger.warning(f"RVC infer error, passthrough: {e}")
            # Fallback: пропустить без конвертации.
            # Сохраняем длину, соответствующую "новому" блоку (overlap исключаем).
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

        # ── VAD gate: если блок тихий — возвращаем тишину без inference ──────
        if self._is_silent(frame):
            # Длина выхода: block_size сэмплов @ output_sr (после ресэмпла)
            out_len = int(self.block_size * self.output_sr / 16000)
            return np.zeros(out_len, dtype=np.float32)

        # Inference
        wav = self._infer_block(frame)

        # Trim overlap с начала выхода (пропорционально)
        trim = int(self.overlap * len(wav) / needed)
        wav = wav[trim:]

        # RMS mix rate — подмес оригинальной огибающей (только если < 1.0)
        if self.rms_mix_rate < 1.0:
            # frame @ 16kHz ↔ wav @ native_sr
            wav = self._apply_rms_mix(
                frame[:self.block_size], wav,
                16000, self._native_sr,
            )

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

        # VAD: короткий хвост — обычно небезусловная тишина, пропускаем gate
        if self._is_silent(frame):
            out_len = int(len(frame) * self.output_sr / 16000)
            return np.zeros(out_len, dtype=np.float32)

        wav = self._infer_block(frame)

        if self.rms_mix_rate < 1.0:
            wav = self._apply_rms_mix(frame, wav, 16000, self._native_sr)

        wav = self._apply_crossfade(wav)
        return self._resample(wav, self._native_sr, self.output_sr)

    def reset(self):
        """Сбросить состояние между фразами."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._sola_buffer = np.zeros(self._crossfade, dtype=np.float32)
        self._cache_pitch = np.zeros(1024, dtype=np.int64)
        self._cache_pitchf = np.zeros(1024, dtype=np.float32)
        if self.extra_frame > 0:
            self._extra_buffer = np.zeros(self.extra_frame, dtype=np.float32)
