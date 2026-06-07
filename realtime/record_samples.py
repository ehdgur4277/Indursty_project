"""
네트워크로 웨이크워드 샘플 녹음 — 서버에서 실행

노트북의 mic_client.py로 오디오를 받아 KWS 학습 데이터로 저장.

사용법:
  [서버] python3 record_samples.py --type pos --count 30
  [노트북] python3 mic_client.py bore.pub --port 9877

  type: pos (웨이크워드 "피식아"), neg (환경 소음/일반 발화)
"""

import argparse
import os
import queue
import socket
import struct
import threading

import numpy as np
import soundfile as sf

SAMPLE_RATE     = 16_000
CLIP_LEN        = 32_000   # 2초
BYTES_PER_CHUNK = 512 * 4  # float32 * CHUNK_SIZE
OUT_DIR         = os.path.join(os.path.dirname(os.path.dirname(__file__)), "KWS", "recordings")


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        data = sock.recv(n - len(buf))
        if not data:
            raise ConnectionError
        buf += data
    return buf


def collect_clip(sock) -> np.ndarray:
    """2초 분량 오디오 수신."""
    chunks = []
    collected = 0
    while collected < CLIP_LEN:
        raw   = recv_exact(sock, BYTES_PER_CHUNK)
        chunk = np.frombuffer(raw, dtype=np.float32)
        chunks.append(chunk)
        collected += len(chunk)
    audio = np.concatenate(chunks)[:CLIP_LEN]
    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",  choices=["pos", "neg"], default="pos",
                        help="pos: 웨이크워드, neg: 환경 소음/일반 발화")
    parser.add_argument("--count", type=int, default=30, help="녹음할 샘플 수")
    parser.add_argument("--port",  type=int, default=9877)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # 기존 파일 번호 이어서
    existing = [f for f in os.listdir(OUT_DIR)
                if f.startswith(f"{args.type}_rec_") and f.endswith(".wav")]
    start_idx = len(existing) + 1

    print(f"[서버] 포트 {args.port} 대기 중...")
    print(f"[노트북] python3 mic_client.py bore.pub --port {args.port}\n")

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    client, addr = server.accept()
    print(f"연결됨: {addr[0]}\n")

    saved = []
    try:
        for i in range(args.count):
            idx = start_idx + i
            fname = f"{args.type}_rec_{idx}.wav"

            if args.type == "pos":
                print(f"[{i+1}/{args.count}] '피식아' 라고 말하세요 → Enter")
            else:
                print(f"[{i+1}/{args.count}] 아무 말 하지 말고 Enter (환경 소음)")

            input()
            print("  녹음 중... (2초)")

            audio = collect_clip(client)
            out_path = os.path.join(OUT_DIR, fname)
            sf.write(out_path, audio, SAMPLE_RATE)
            saved.append(out_path)
            print(f"  저장: {fname}")

    except (KeyboardInterrupt, ConnectionError):
        pass
    finally:
        client.close()
        server.close()

    print(f"\n완료! {len(saved)}개 저장 → KWS/recordings/")
    print("\n이제 재학습:")
    print("  cd KWS && python3 train_oww.py")


if __name__ == "__main__":
    main()
