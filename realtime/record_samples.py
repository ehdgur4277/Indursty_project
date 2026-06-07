"""
네트워크로 KWS 학습 샘플 녹음

[연구실 컴퓨터] python3 record_samples.py --type pos --count 30
[연구실 컴퓨터] ~/bore local 9877 --to bore.pub  (별도 터미널)
[노트북]        python3 mic_client.py bore.pub --port <포트번호>

type: pos → "피식아" 녹음 / neg → 환경 소음 녹음
"""

import argparse
import os
import socket
import time

import numpy as np
import soundfile as sf

SAMPLE_RATE     = 16_000
CLIP_LEN        = 32_000        # 2초
BYTES_PER_CHUNK = 512 * 4       # float32 * CHUNK_SIZE
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "KWS", "recordings")


def recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        data = sock.recv(n - len(buf))
        if not data:
            raise ConnectionError("연결 끊김")
        buf += data
    return buf


def flush(sock, seconds: float):
    """지정 시간만큼 들어오는 오디오를 버림."""
    end = time.time() + seconds
    while time.time() < end:
        try:
            recv_exact(sock, BYTES_PER_CHUNK)
        except Exception:
            break


def collect_clip(sock) -> np.ndarray:
    """2초 분량 오디오 수신."""
    chunks, collected = [], 0
    while collected < CLIP_LEN:
        raw   = recv_exact(sock, BYTES_PER_CHUNK)
        chunk = np.frombuffer(raw, dtype=np.float32)
        chunks.append(chunk)
        collected += len(chunk)
    return np.concatenate(chunks)[:CLIP_LEN].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",  choices=["pos", "neg"], default="pos")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--port",  type=int, default=9877)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    existing  = [f for f in os.listdir(OUT_DIR)
                 if f.startswith(f"{args.type}_rec_") and f.endswith(".wav")]
    start_idx = len(existing) + 1

    # ── 서버 소켓 오픈 ────────────────────────────────────────────
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)

    print(f"\n{'='*45}")
    print(f"[연구실 컴퓨터] 포트 {args.port} 대기 중...")
    print(f"[연구실 컴퓨터] 별도 터미널: ~/bore local {args.port} --to bore.pub")
    print(f"[노트북]        python3 mic_client.py bore.pub --port <번호>")
    print(f"{'='*45}\n")

    client, addr = server.accept()
    print(f"✅ 노트북 연결됨 ({addr[0]})\n")
    time.sleep(0.5)

    label = "'피식아' 라고 말하세요" if args.type == "pos" else "아무 말도 하지 마세요 (소음 녹음)"
    saved = []

    try:
        for i in range(args.count):
            idx   = start_idx + i
            fname = f"{args.type}_rec_{idx}.wav"

            print(f"[{i+1:2d}/{args.count}] {label}")
            for c in [3, 2, 1]:
                print(f"       {c}...", end="\r", flush=True)
                time.sleep(1)
            print(f"       🔴 녹음!      ", flush=True)

            flush(client, 0.1)          # 카운트다운 중 들어온 오디오 버림
            audio = collect_clip(client)

            out_path = os.path.join(OUT_DIR, fname)
            sf.write(out_path, audio, SAMPLE_RATE)
            saved.append(out_path)
            print(f"       ✅ 저장: {fname}\n")
            time.sleep(0.3)

    except (KeyboardInterrupt, ConnectionError) as e:
        print(f"\n중단: {e}")
    finally:
        client.close()
        server.close()

    print(f"\n{'='*45}")
    print(f"완료! {len(saved)}개 저장 → KWS/recordings/")
    print(f"\n[연구실 컴퓨터] 재학습:")
    print(f"  cd /data/dhpark/Indursty_project/KWS")
    print(f"  python3 train_oww.py")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
