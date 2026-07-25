"""
rvc_py/bulk_convert.py — Bulk offline RVC conversion с ре-энкодингом в WAV/MP3.

Гоняет папку (или один файл) через RVC. Модели кешируются между файлами
(HuBERT/RVC/RMVPE грузятся один раз на весь батч, не на каждый файл) —
поэтому обработка сотни реплик занимает секунды, а не минуты.

Использование (CLI, после pip install):
    rvc-bulk --input ./voice_lines --output ./converted \\
             --model freddy.pth --index freddy.index --pitch -2 \\
             --format wav --sample-rate 22050 --bit-depth 16

    # Готовый пресет под движок игры:
    rvc-bulk --input ./voice_lines --output ./converted \\
             --model freddy.pth --preset game_modern

    # MP3 (нужен ffmpeg в PATH + pip install pydub):
    rvc-bulk --input ./voice_lines --output ./converted \\
             --model freddy.pth --format mp3 --bitrate 128

Использование (из кода):
    from rvc_py.bulk_convert import bulk_convert

    bulk_convert(
        "./voice_lines", "./converted", "freddy.pth",
        pitch_shift=-2, out_format="wav", out_sample_rate=22050,
    )

Пресеты (--preset):
    game_legacy  — WAV, 22050Hz, mono, 16-bit PCM   (RPG Maker и т.п.)
    game_modern  — WAV, 44100Hz, mono, 16-bit PCM   (Unity/Unreal/Godot)
    web_mp3      — MP3, 44100Hz, mono, 128kbps       (веб/стриминг)
"""
from __future__ import annotations

import os
import glob
import logging

import numpy as np
import soundfile as sf

logger = logging.getLogger("rvc_bulk")

SUPPORTED_INPUT_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}

PRESETS = {
    "game_legacy": {"format": "wav", "sample_rate": 22050, "bit_depth": 16},
    "game_modern": {"format": "wav", "sample_rate": 44100, "bit_depth": 16},
    "web_mp3":     {"format": "mp3", "sample_rate": 44100, "bitrate": 128},
    # Source принимает WAV ТОЛЬКО на 11025 / 22050 / 44100 Hz — любая другая
    # частота (например 48000) просто не заиграется в движке.
    # 44100/16-bit/mono — стандарт для озвучки персонажей/lip-sync в HL2 и др.
    # https://developer.valvesoftware.com/wiki/WAV
    "source_engine": {"format": "wav", "sample_rate": 44100, "bit_depth": 16},
}


def _find_input_files(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    files = []
    for ext in SUPPORTED_INPUT_EXT:
        files.extend(glob.glob(os.path.join(input_path, "**", f"*{ext}"), recursive=True))
    return sorted(files)


def _resample_and_bitdepth(
    audio: np.ndarray, sr: int, target_sr: int, bit_depth: int
) -> tuple[np.ndarray, str]:
    """Ресэмпл + подготовка под нужную битность. Возвращает (audio, soundfile_subtype)."""
    if sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)

    subtype_map = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}
    subtype = subtype_map.get(bit_depth, "PCM_16")
    return audio, subtype


def _export_mp3(audio: np.ndarray, sr: int, out_path: str, bitrate: int):
    """Экспорт в MP3 через pydub (требует ffmpeg в PATH)."""
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise ImportError(
            "Для экспорта в MP3 нужен pydub + ffmpeg:\n"
            "  pip install pydub\n"
            "  # ffmpeg должен быть в PATH: apt install ffmpeg (linux) "
            "/ choco install ffmpeg (windows)"
        ) from e

    int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    seg = AudioSegment(int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    seg.export(out_path, format="mp3", bitrate=f"{bitrate}k")


def _auto_gain_db(audio: np.ndarray, target_peak_db: float = -1.0) -> float:
    """
    Вычисляет дБ усиления/ослабления, чтобы довести пик сигнала до target_peak_db.
    Полезно для батч-конверсии — приводит все реплики к одному уровню
    громкости автоматически, без ручного подбора --gain для каждого файла.
    target_peak_db=-1.0 — оставляет 1 дБ запаса от 0 dBFS (защита от
    intersample-пиков при последующем MP3-энкодинге).
    """
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak < 1e-6:
        return 0.0
    target_linear = 10 ** (target_peak_db / 20.0)
    return float(20 * np.log10(target_linear / peak))


def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Линейное усиление/ослабление в дБ. Гасит в [-1, 1] чтобы избежать
    заворачивания (wraparound) при конвертации в int16, если гейн слишком большой.
    gain_db=0.0 — выключено, никакого доп. вычисления.
    """
    if gain_db == 0.0:
        return audio
    gain_linear = 10 ** (gain_db / 20.0)
    out = (audio * gain_linear).astype(np.float32)
    peak = np.abs(out).max() if len(out) else 0.0
    if peak > 1.0:
        logger.warning(
            f"Гейн {gain_db:+.1f}дБ даёт клиппинг (пик={peak:.2f}), ограничиваю до 0 dBFS"
        )
        out = np.clip(out, -1.0, 1.0)
    return out


def convert_file(
    in_path: str,
    out_path: str,
    rvc_model_path: str,
    device: str = "cuda",
    index_path: str | None = None,
    index_rate: float = 0.5,
    pitch_shift: int = 0,
    f0_method: str = "rmvpe",
    out_format: str = "wav",
    out_sample_rate: int = 40000,
    bit_depth: int = 16,
    mp3_bitrate: int = 128,
    filter_radius: float = 0.03,
    rms_mix_rate: float = 1.0,
    protect: float = 0.5,
    gain_db: float = 0.0,
    auto_gain: bool = False,
    target_peak_db: float = -1.0,
) -> None:
    """Конвертирует один файл через RVC и сохраняет в нужном формате/параметрах."""
    from rvc_py.rvc_infer import rvc_infer

    wav, sr = sf.read(in_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # стерео → моно перед RVC

    out_wav, out_sr = rvc_infer(
        wav, sr, rvc_model_path,
        device=device, index_path=index_path, index_rate=index_rate,
        pitch_shift=pitch_shift, f0_method=f0_method,
        filter_radius=filter_radius, rms_mix_rate=rms_mix_rate, protect=protect,
    )

    total_gain_db = gain_db
    if auto_gain:
        total_gain_db += _auto_gain_db(out_wav, target_peak_db)

    out_wav = _apply_gain(out_wav, total_gain_db)

    final_audio, subtype = _resample_and_bitdepth(out_wav, out_sr, out_sample_rate, bit_depth)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if out_format == "mp3":
        _export_mp3(final_audio, out_sample_rate, out_path, mp3_bitrate)
    else:
        sf.write(out_path, final_audio, out_sample_rate, subtype=subtype)


def bulk_convert(
    input_path: str,
    output_dir: str,
    rvc_model_path: str,
    device: str = "cuda",
    index_path: str | None = None,
    index_rate: float = 0.5,
    pitch_shift: int = 0,
    f0_method: str = "rmvpe",
    out_format: str = "wav",
    out_sample_rate: int = 40000,
    bit_depth: int = 16,
    mp3_bitrate: int = 128,
    preserve_structure: bool = True,
    filter_radius: float = 0.03,
    rms_mix_rate: float = 1.0,
    protect: float = 0.5,
    gain_db: float = 0.0,
    auto_gain: bool = False,
    target_peak_db: float = -1.0,
) -> list[str]:
    """
    Гоняет все файлы из input_path (файл или папка, рекурсивно) через RVC.
    Модели (HuBERT/RVC/RMVPE) кешируются между файлами — грузятся один раз
    за весь вызов, не на каждый файл.

    Returns:
        Список путей к успешно созданным выходным файлам.
    """
    from tqdm import tqdm

    files = _find_input_files(input_path)
    if not files:
        logger.warning(f"Не найдено аудиофайлов в: {input_path}")
        return []

    logger.info(f"Найдено {len(files)} файлов, начинаю конвертацию...")
    ext = ".mp3" if out_format == "mp3" else ".wav"
    outputs: list[str] = []

    input_root = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)

    for f in tqdm(files, desc="RVC bulk convert", unit="file"):
        if preserve_structure and os.path.isdir(input_path):
            rel = os.path.relpath(f, input_root)
        else:
            rel = os.path.basename(f)
        out_name = os.path.splitext(rel)[0] + ext
        out_path = os.path.join(output_dir, out_name)

        try:
            convert_file(
                f, out_path, rvc_model_path,
                device=device, index_path=index_path, index_rate=index_rate,
                pitch_shift=pitch_shift, f0_method=f0_method,
                out_format=out_format, out_sample_rate=out_sample_rate,
                bit_depth=bit_depth, mp3_bitrate=mp3_bitrate,
                filter_radius=filter_radius, rms_mix_rate=rms_mix_rate, protect=protect,
                gain_db=gain_db, auto_gain=auto_gain, target_peak_db=target_peak_db,
            )
            outputs.append(out_path)
        except Exception as e:
            logger.error(f"Ошибка на {f}: {e}")

    logger.info(f"Готово: {len(outputs)}/{len(files)} файлов → {output_dir}")
    return outputs


def main():
    """CLI точка входа: rvc-bulk"""
    import argparse

    p = argparse.ArgumentParser(description="Bulk RVC conversion с ре-энкодингом в WAV/MP3")
    p.add_argument("--input", required=True, help="Файл или папка с исходным аудио")
    p.add_argument("--output", required=True, help="Папка для результатов")
    p.add_argument("--model", required=True, help="Путь к .pth RVC модели")
    p.add_argument("--index", default=None, help="Путь к .index (опционально)")
    p.add_argument("--index-rate", type=float, default=0.5)
    p.add_argument("--pitch", type=int, default=0, help="Сдвиг питча в полутонах")
    p.add_argument("--f0", default="rmvpe", choices=["rmvpe"])
    p.add_argument("--device", default="cuda")

    p.add_argument(
        "--preset", default=None, choices=list(PRESETS.keys()),
        help="Готовый набор параметров вывода: game_legacy / game_modern / web_mp3",
    )
    p.add_argument("--format", default="wav", choices=["wav", "mp3"])
    p.add_argument("--sample-rate", type=int, default=40000)
    p.add_argument("--bit-depth", type=int, default=16, choices=[16, 24, 32])
    p.add_argument("--bitrate", type=int, default=128, help="MP3 битрейт в kbps")
    p.add_argument(
        "--flat", action="store_true",
        help="Не сохранять структуру подпапок, всё в output/ плоско",
    )

    p.add_argument(
        "--filter-radius", type=float, default=0.03,
        help="Порог уверенности RMVPE (thred). Не медианный фильтр — см. rvc_infer.py. "
             "0.03 = дефолт Applio и прежнее захардкоженное поведение",
    )
    p.add_argument(
        "--rms-mix-rate", type=float, default=1.0,
        help="Подмес громкостной огибающей. 1.0 = выключено",
    )
    p.add_argument(
        "--protect", type=float, default=0.5,
        help="Защита глухих участков. 0.5 = выключено, 0.33 = дефолт Applio",
    )
    p.add_argument(
        "--gain", type=float, default=0.0,
        help="Доп. усиление в дБ (+6 если тихо для Source). Автоограничение при клиппинге",
    )
    p.add_argument(
        "--auto-gain", action="store_true",
        help="Автоматически выровнять пик каждого файла до --target-peak (складывается с --gain)",
    )
    p.add_argument(
        "--target-peak", type=float, default=-1.0,
        help="Целевой пик в dBFS для --auto-gain (дефолт -1.0)",
    )

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = dict(
        out_format=args.format, out_sample_rate=args.sample_rate,
        bit_depth=args.bit_depth, mp3_bitrate=args.bitrate,
    )
    if args.preset:
        preset = PRESETS[args.preset]
        cfg["out_format"] = preset.get("format", cfg["out_format"])
        cfg["out_sample_rate"] = preset.get("sample_rate", cfg["out_sample_rate"])
        cfg["bit_depth"] = preset.get("bit_depth", cfg["bit_depth"])
        cfg["mp3_bitrate"] = preset.get("bitrate", cfg["mp3_bitrate"])
        logger.info(f"Пресет '{args.preset}': {cfg}")

    bulk_convert(
        args.input, args.output, args.model,
        device=args.device, index_path=args.index, index_rate=args.index_rate,
        pitch_shift=args.pitch, f0_method=args.f0,
        preserve_structure=not args.flat,
        filter_radius=args.filter_radius, rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        gain_db=args.gain,
        auto_gain=args.auto_gain, target_peak_db=args.target_peak,
        **cfg,
    )


if __name__ == "__main__":
    main()
