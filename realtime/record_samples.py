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
_REC_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "KWS", "recordings")


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        data = sock.recv(n - len(buf))
        if not data:
            raise ConnectionError
        buf += data
    return buf


def flush_buffer(sock):
    """소켓 버퍼에 쌓인 오디오를 버림 (Enter 이전 오디오 제거)."""
    sock.setblocking(False)
    try:
        while True:
            sock.recv(65536)
    except BlockingIOError:
        pass
    finally:
        sock.setblocking(True)


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

    OUT_DIR = os.path.join(_REC_ROOT, args.type)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 기존 파일 번호 이어서
    existing = [f for f in os.listdir(OUT_DIR) if f.endswith(".wav")]
    start_idx = len(existing) + 1

    print(f"[연구실 컴퓨터] 포트 {args.port} 대기 중...")
    print(f"[연구실 컴퓨터] 별도 터미널: ~/bore local {args.port} --to bore.pub")
    print(f"[노트북]        python3 mic_client.py bore.pub --port <번호>\n")

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
            fname = f"rec_{idx}.wav"

            if args.type == "pos":
                print(f"[{i+1}/{args.count}] [연구실 컴퓨터] Enter → [노트북] '피식아' 말하기")
            else:
                print(f"[{i+1}/{args.count}] [연구실 컴퓨터] Enter → 아무 말 하지 말기 (소음 녹음)")

            input()
            flush_buffer(client)        # Enter 이전에 쌓인 오디오 버림
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
