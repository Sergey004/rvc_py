"""CLI entry points для rvc-py."""
import argparse
import sys


def cmd_infer():
    """rvc-infer --model model.pth --input in.wav --output out.wav [--pitch 0]"""
    p = argparse.ArgumentParser(description="RVC offline inference")
    p.add_argument("--model",   required=True,  help="Путь к .pth модели")
    p.add_argument("--index",   default=None,   help="Путь к .index (опционально)")
    p.add_argument("--input",   required=True,  help="Входной WAV/MP3")
    p.add_argument("--output",  required=True,  help="Выходной WAV")
    p.add_argument("--pitch",   type=int, default=0, help="Сдвиг питча в полутонах")
    p.add_argument("--f0",      default="rmvpe",
                   choices=["rmvpe", "fcpe", "harvest", "crepe"],
                   help="Метод экстракции F0")
    p.add_argument("--device",  default="cuda", help="cuda / cpu / cuda:1")
    args = p.parse_args()

    from rvc_py.rvc_model import RVCModel
    import soundfile as sf
    import numpy as np

    print(f"[rvc-infer] Загружаю модель: {args.model}")
    model = RVCModel(args.model, index_path=args.index, device=args.device)

    print(f"[rvc-infer] Читаю: {args.input}")
    audio, sr = sf.read(args.input, dtype="float32", always_2d=False)

    print(f"[rvc-infer] Конвертирую (pitch={args.pitch:+d}, f0={args.f0})...")
    out = model.infer(audio, sr, f0_up_key=args.pitch, f0_method=args.f0)

    sf.write(args.output, out, 40000)  # RVC всегда 40k на выходе
    print(f"[rvc-infer] Готово → {args.output}")


def cmd_download():
    """rvc-download [--models rmvpe contentvec] [--dir ~/.cache/rvc]"""
    p = argparse.ArgumentParser(description="Скачать веса моделей RVC")
    p.add_argument(
        "--models", nargs="+",
        default=["rmvpe", "contentvec"],
        choices=["rmvpe", "contentvec", "fcpe", "hubert"],
        help="Какие модели скачать"
    )
    p.add_argument("--dir", default=None, help="Куда сохранить (по умолчанию ~/.cache/rvc)")
    args = p.parse_args()

    from rvc_py.download_models import download
    download(models=args.models, cache_dir=args.dir)


def cmd_devices():
    """rvc-devices — список доступных аудиоустройств"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\n=== Аудиоустройства ===")
        for i, d in enumerate(devices):
            tag = ""
            if d["max_input_channels"] > 0:  tag += " [IN]"
            if d["max_output_channels"] > 0: tag += " [OUT]"
            print(f"  {i:2d}: {d['name']}{tag}  @ {int(d['default_samplerate'])}Hz")
    except ImportError:
        print("sounddevice не установлен: pip install sounddevice")
        sys.exit(1)


def cmd_serve():
    """rvc-serve --model model.pth [--index model.index] [--port 8788]

    Client/server real-time voice conversion через WebSocket.
    Открой rvc_py/web_client/index.html в браузере и подключись к серверу
    (или используй любой WS-клиент, говорящий на протоколе из server.py).
    """
    from rvc_py.server import main as server_main
    server_main()


def cmd_realtime():
    """rvc-rt --model model.pth [--pitch 0] [--input 1] [--output 3] [--block 1024]"""
    p = argparse.ArgumentParser(description="RVC real-time voice changer")
    p.add_argument("--model",   required=True)
    p.add_argument("--index",   default=None)
    p.add_argument("--pitch",   type=int, default=0)
    p.add_argument("--input",   type=int, default=None, help="ID входного устройства")
    p.add_argument("--output",  type=int, default=None, help="ID выходного устройства")
    p.add_argument("--block",   type=int, default=1024, help="Размер блока @ 16kHz")
    p.add_argument("--f0",      default="rmvpe", choices=["rmvpe", "fcpe", "harvest"])
    p.add_argument("--no-vad",  action="store_true", help="Отключить Silero VAD")
    p.add_argument("--no-rnn",  action="store_true", help="Отключить RNNoise")
    args = p.parse_args()

    # Импортируем RT модуль только здесь — sounddevice опциональный
    try:
        from rvc_py.realtime import RealTimeRVC
    except ImportError as e:
        print(f"Для realtime нужен sounddevice: pip install sounddevice\n{e}")
        sys.exit(1)

    from rvc_py.rvc_model import RVCModel
    model = RVCModel(args.model, index_path=args.index)

    rt = RealTimeRVC(
        rvc_model=model,
        f0_up_key=args.pitch,
        f0_method=args.f0,
        block_size=args.block,
        input_device=args.input,
        output_device=args.output,
        use_vad=not args.no_vad,
        use_rnnoise=not args.no_rnn,
    )

    rt.start()
    print(f"\n🎤 Говори в микрофон. Pitch: {args.pitch:+d}")
    print("Команды: +N / -N = сдвиг питча, q = выход\n")
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd.lstrip("+-").isdigit():
                rt.set_pitch(int(cmd))
                print(f"  pitch → {rt.f0_up_key:+d}")
    except KeyboardInterrupt:
        pass
    finally:
        rt.stop()