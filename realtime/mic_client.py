"""
노트북 마이크 → 서버 TCP 스트리밍 클라이언트

이 파일은 **노트북**에서 실행합니다.
서버에서 rt_pipeline.py --network를 먼저 실행한 뒤 이 스크립트를 실행하세요.

설치 (노트북):
    pip install sounddevice numpy

실행:
    python3 mic_client.py <서버IP>
    python3 mic_client.py <서버IP> --port 9876
    python3 mic_client.py <서버IP> --device     # 마이크 장치 목록 출력
"""

import argparse
import socket
import sys
import threading
import time

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SIZE  = 512       # 서버와 동일해야 함
PORT        = 9876


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())
    sys.exit(0)


def _recv_loop(sock: socket.socket):
    """서버에서 오는 결과 메시지를 수신해서 출력하는 스레드."""
    buf = ""
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                break
            buf += data.decode("utf-8", errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if line.strip():
                    print(line)
        except OSError:
            break


def stream_mic(host: str, port: int, device=None):
    import sounddevice as sd

    print(f"서버 {host}:{port} 연결 중...")
    with socket.socket() as sock:
        sock.connect((host, port))
        print(f"연결됨. 마이크 스트리밍 시작 (Ctrl+C로 종료)\n")

        # 서버 결과 수신 스레드
        recv_thread = threading.Thread(target=_recv_loop, args=(sock,), daemon=True)
        recv_thread.start()

        def callback(indata, frames, time_info, status):
            audio = indata[:, 0].astype(np.float32)
            try:
                sock.sendall(audio.tobytes())
            except OSError:
                raise sd.CallbackStop()

        stream_kwargs = dict(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=callback,
        )
        if device is not None:
            stream_kwargs["device"] = device

        with sd.InputStream(**stream_kwargs):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n종료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="노트북 마이크를 서버로 스트리밍",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "예시:\n"
            "  python3 mic_client.py 192.168.0.10\n"
            "  python3 mic_client.py 192.168.0.10 --port 9876\n"
            "  python3 mic_client.py --device          # 마이크 목록 확인"
        ),
    )
    parser.add_argument("host",     nargs="?", default=None, help="서버 IP 주소")
    parser.add_argument("--port",   type=int,  default=PORT, help=f"서버 포트 (기본 {PORT})")
    parser.add_argument("--device", nargs="?", const=True,   help="마이크 장치 번호, 또는 플래그만 쓰면 목록 출력")
    args = parser.parse_args()

    if args.device is True:
        list_devices()

    if args.host is None:
        parser.error("서버 IP를 입력하세요. 예: python3 mic_client.py 192.168.0.10")

    device = int(args.device) if isinstance(args.device, str) else None
    stream_mic(args.host, args.port, device=device)
