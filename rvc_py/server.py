"""
rvc_py/server.py — WebSocket сервер для client/server real-time voice conversion.

Оборачивает RVCStreamer в простой бинарно/JSON протокол, который может
говорить любой клиент: браузер (WebAudio + WebSocket), Python, что угодно.

Протокол
--------
Клиент → Сервер, JSON (текстовый фрейм), первое сообщение:
    {
        "type": "init",
        "sample_rate": 48000,        # SR клиента (обычно 48000 в браузере)
        "pitch_shift": 0,             # полутоны
        "f0_method": "rmvpe",         # rmvpe | fcpe | harvest
        "block_size": 2048,           # размер блока @ 16kHz (латентность/качество)
        "index_rate": 0.0,
        "model": "optional_model_id" # только если сервер разрешил --allow-client-model
    }

Клиент → Сервер, бинарный фрейм:
    raw Float32LE PCM, mono, на sample_rate из init

Клиент → Сервер, JSON:
    {"type": "flush"}                  — конец фразы, выдать остаток буфера
    {"type": "set_pitch", "value": -5}  — сменить питч на лету
    {"type": "reset"}                   — сбросить состояние стримера

Сервер → Клиент, бинарный фрейм:
    raw Float32LE PCM, mono, на sample_rate клиента (уже сконвертировано RVC)

Сервер → Клиент, JSON:
    {"type": "ready"}
    {"type": "error", "message": "..."}

Запуск
------
    rvc-serve --model freddy.pth --index freddy.index --port 8788

    # Разрешить клиенту указывать свою модель (ОСТОРОЖНО — доверяй только
    # клиентам в своей сети, путь к файлу приходит от клиента):
    rvc-serve --model freddy.pth --allow-client-model
"""
from __future__ import annotations

import asyncio
import json
import logging

import numpy as np

logger = logging.getLogger("rvc_server")

# Сериализует GPU inference между сессиями — предотвращает состояние гонки
# при параллельных клиентах на одной GPU. Буфер каждой сессии независим,
# инференс идёт по очереди.
_GLOBAL_INFER_LOCK = asyncio.Lock()


class RVCSession:
    """Обёртка над RVCStreamer для одного WebSocket-соединения."""

    def __init__(self, ws, config: dict):
        from .realtime import RVCStreamer

        self.ws = ws
        sr = int(config.get("sample_rate", 48000))

        self.streamer = RVCStreamer(
            rvc_model_path=config["model_path"],
            device=config.get("device", "cuda"),
            f0_up_key=int(config.get("pitch_shift", 0)),
            f0_method=config.get("f0_method", "rmvpe"),
            index_path=config.get("index_path"),
            index_rate=float(config.get("index_rate", 0.0)),
            block_size=int(config.get("block_size", 2048)),
            overlap=int(config.get("overlap", 512)),
            crossfade=int(config.get("crossfade", 256)),
            input_sr=sr,
            output_sr=sr,
        )
        self.sample_rate = sr

    async def push(self, pcm_bytes: bytes):
        audio = np.frombuffer(pcm_bytes, dtype=np.float32)
        loop = asyncio.get_event_loop()
        async with _GLOBAL_INFER_LOCK:
            out = await loop.run_in_executor(None, self.streamer.push, audio)
        if out is not None and len(out) > 0:
            await self.ws.send(out.astype(np.float32).tobytes())

    async def flush(self):
        loop = asyncio.get_event_loop()
        async with _GLOBAL_INFER_LOCK:
            out = await loop.run_in_executor(None, self.streamer.flush)
        if out is not None and len(out) > 0:
            await self.ws.send(out.astype(np.float32).tobytes())

    def set_pitch(self, value: int):
        self.streamer.f0_up_key = value

    def reset(self):
        self.streamer.reset()


class RVCServer:
    """
    WebSocket сервер, принимающий несколько одновременных клиентов.
    Каждое соединение получает свой RVCSession (свой буфер),
    но веса модели общие — кешируются в rvc_infer._get_or_create_models.
    """

    def __init__(
        self,
        default_model_path: str | None = None,
        default_index_path: str | None = None,
        device: str = "cuda",
        host: str = "0.0.0.0",
        port: int = 8788,
        allow_client_model: bool = False,
    ):
        self.default_model_path = default_model_path
        self.default_index_path = default_index_path
        self.device = device
        self.host = host
        self.port = port
        self.allow_client_model = allow_client_model

    async def _handle(self, ws):
        import websockets

        session: RVCSession | None = None
        remote = getattr(ws, "remote_address", "?")
        logger.info(f"[RVC Server] client connected: {remote}")

        try:
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    if session is None:
                        continue
                    try:
                        await session.push(bytes(message))
                    except Exception as e:
                        logger.error(f"push error: {e}")
                        try:
                            await ws.send(json.dumps({"type": "error", "message": str(e)}))
                        except Exception:
                            pass
                    continue

                # JSON control frame
                try:
                    data = json.loads(message)
                except Exception:
                    continue

                mtype = data.get("type")

                if mtype == "init":
                    cfg = dict(data)
                    if self.allow_client_model and cfg.get("model"):
                        cfg["model_path"] = cfg["model"]
                    else:
                        cfg["model_path"] = self.default_model_path
                        cfg["index_path"] = self.default_index_path
                    cfg.setdefault("device", self.device)

                    if not cfg.get("model_path"):
                        await ws.send(json.dumps({
                            "type": "error",
                            "message": "no model configured on server (start with --model)",
                        }))
                        continue

                    try:
                        session = RVCSession(ws, cfg)
                        await ws.send(json.dumps({
                            "type": "ready",
                            "sample_rate": session.sample_rate,
                        }))
                        logger.info(
                            f"[RVC Server] session ready: {remote} "
                            f"sr={session.sample_rate} pitch={session.streamer.f0_up_key:+d}"
                        )
                    except Exception as e:
                        logger.error(f"init error: {e}")
                        await ws.send(json.dumps({"type": "error", "message": str(e)}))

                elif mtype == "flush":
                    if session:
                        await session.flush()

                elif mtype == "set_pitch":
                    if session:
                        session.set_pitch(int(data.get("value", 0)))

                elif mtype == "reset":
                    if session:
                        session.reset()

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"[RVC Server] connection error [{remote}]: {e}", exc_info=True)
        finally:
            logger.info(f"[RVC Server] client disconnected: {remote}")

    async def serve_forever(self):
        import websockets

        logger.info(f"[RVC Server] listening on ws://{self.host}:{self.port}")
        logger.info(f"[RVC Server] default model: {self.default_model_path}")
        async with websockets.serve(
            self._handle, self.host, self.port,
            max_size=20 * 1024 * 1024,
            ping_interval=20, ping_timeout=30,
        ):
            await asyncio.Future()

    def run(self):
        asyncio.run(self.serve_forever())


def main():
    """Точка входа для `rvc-serve` CLI команды."""
    import argparse

    p = argparse.ArgumentParser(description="RVC WebSocket real-time server (client/server)")
    p.add_argument("--model", required=True, help="Путь к .pth модели по умолчанию")
    p.add_argument("--index", default=None, help="Путь к .index (опционально)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8788)
    p.add_argument(
        "--allow-client-model", action="store_true",
        help="Разрешить клиенту указывать свой model_path через init "
             "(ОСТОРОЖНО: доверяй только клиентам в своей сети)",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    server = RVCServer(
        default_model_path=args.model,
        default_index_path=args.index,
        device=args.device,
        host=args.host,
        port=args.port,
        allow_client_model=args.allow_client_model,
    )
    server.run()


if __name__ == "__main__":
    main()
