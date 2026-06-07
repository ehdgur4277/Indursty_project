"""마이크/파일 오디오 캡처 모듈 — sounddevice 기반 청크 스트림"""
import os
import queue
import sys
import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
CHUNK_SIZE  = 512   # ~32 ms @ 16 kHz


class AudioStream:
    """실시간 마이크 입력 스트림."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size  = chunk_size
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time, status):
        self._q.put(indata[:, 0].copy().astype(np.float32))

    def start(self):
        input_devs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        if not input_devs:
            raise RuntimeError(
                "마이크 입력 장치를 찾을 수 없습니다.\n"
                "  Docker 환경에서는 파일 입력 모드를 사용하세요:\n"
                "    python3 rt_pipeline.py --file <경로> [--skip_kws]"
            )
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float = 1.0) -> np.ndarray:
        """큐에서 청크 하나를 꺼냄. timeout 초 내에 없으면 queue.Empty 발생."""
        return self._q.get(timeout=timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class FileStream:
    """
    오디오 파일을 CHUNK_SIZE 단위로 서빙 — 마이크 없는 환경 테스트용.
    AudioStream과 동일한 인터페이스를 제공.
    파일이 끝나면 read()가 queue.Empty를 발생시킴.
    """

    def __init__(self, file_path: str, sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE):
        self.file_path   = file_path
        self.sample_rate = sample_rate
        self.chunk_size  = chunk_size
        self._q: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=256)
        self._thread: "threading.Thread | None"    = None
        self._stop_event = threading.Event()

    def _worker(self):
        _whisper = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Whisper")
        if _whisper not in sys.path:
            sys.path.insert(0, _whisper)
        from audio_loader import load_audio

        audio  = load_audio(self.file_path)
        offset = 0
        while offset < len(audio) and not self._stop_event.is_set():
            chunk = audio[offset: offset + self.chunk_size]
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
            self._q.put(chunk.astype(np.float32))
            offset += self.chunk_size
        self._q.put(None)   # 파일 끝 신호

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def read(self, timeout: float = 1.0) -> np.ndarray:
        """청크를 꺼냄. 파일이 끝나면 queue.Empty 발생."""
        item = self._q.get(timeout=timeout)
        if item is None:
            raise queue.Empty("파일 끝")
        return item

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class NetworkStream:
    """
    TCP 소켓으로 원격 마이크 오디오를 수신하는 스트림.
    노트북의 mic_client.py가 연결하면 그 오디오를 파이프라인에 공급.

    프로토콜: float32 * CHUNK_SIZE = 2048 bytes 고정 청크 (헤더 없음)
    """

    _BYTES_PER_CHUNK = CHUNK_SIZE * 4   # float32 = 4 bytes

    def __init__(self, host: str = "0.0.0.0", port: int = 9876,
                 chunk_size: int = CHUNK_SIZE, sample_rate: int = SAMPLE_RATE):
        self.host        = host
        self.port        = port
        self.chunk_size  = chunk_size
        self.sample_rate = sample_rate
        self._q: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=512)
        self._server_sock = None
        self._client_sock = None
        self._thread: "threading.Thread | None" = None
        self._stop_event  = threading.Event()

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            data = self._client_sock.recv(n - len(buf))
            if not data:
                raise ConnectionError("클라이언트 연결 종료")
            buf += data
        return buf

    def _worker(self):
        try:
            while not self._stop_event.is_set():
                raw   = self._recv_exact(self._BYTES_PER_CHUNK)
                chunk = np.frombuffer(raw, dtype=np.float32).copy()
                self._q.put(chunk)
        except (ConnectionError, OSError):
            pass
        finally:
            self._q.put(None)   # 종료 신호

    def send(self, message: str):
        """노트북으로 텍스트 메시지 전송 (줄바꿈 구분)."""
        if self._client_sock:
            try:
                self._client_sock.sendall((message + "\n").encode("utf-8"))
            except OSError:
                pass

    def start(self):
        import socket
        self._server_sock = socket.socket()
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)
        print(f"[네트워크] TCP 대기 중 — 포트 {self.port}")
        print(f"[네트워크] 노트북에서 실행:\n"
              f"    python3 mic_client.py <서버IP> --port {self.port}\n")
        self._client_sock, addr = self._server_sock.accept()
        print(f"[네트워크] {addr[0]}:{addr[1]} 연결됨\n")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            if self._client_sock:
                self._client_sock.close()
            if self._server_sock:
                self._server_sock.close()
        except OSError:
            pass

    def read(self, timeout: float = 1.0) -> np.ndarray:
        item = self._q.get(timeout=timeout)
        if item is None:
            raise queue.Empty("클라이언트 연결 종료")
        return item

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
