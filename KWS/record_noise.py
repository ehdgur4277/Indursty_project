"""
환경 소음 녹음 스크립트 — 노트북에서 실행

KWS 오감지 해결을 위해 현재 환경 배경음을 negative 샘플로 녹음.
녹음된 파일을 서버 KWS/recordings/ 폴더에 복사 후 train_oww.py 재실행.

설치: pip install sounddevice soundfile numpy
실행: python3 record_noise.py
"""

import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000
DURATION    = 30          # 녹음 시간 (초)
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))


def record(filename: str, duration: int):
    print(f"[녹음] {duration}초간 배경음 녹음 시작 — 아무 말도 하지 마세요...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("  녹음 중...")

    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()

    out_path = os.path.join(OUT_DIR, filename)
    sf.write(out_path, audio, SAMPLE_RATE)
    print(f"[완료] {out_path} 저장됨\n")
    return out_path


if __name__ == "__main__":
    # 기존 neg_env 파일 번호 이어서
    existing = [f for f in os.listdir(OUT_DIR)
                if f.startswith("neg_env_") and f.endswith(".wav")]
    idx = len(existing) + 1

    files = []
    for i in range(3):
        fname = f"neg_env_{idx + i}.wav"
        path  = record(fname, DURATION)
        files.append(path)
        if i < 2:
            input("다음 녹음 준비되면 Enter...")

    print("=" * 40)
    print("녹음 완료! 아래 파일들을 서버의 KWS/recordings/ 폴더에 복사하세요:")
    for f in files:
        print(f"  {f}")
    print("\n복사 후 서버에서 재학습:")
    print("  cd KWS && python3 train_oww.py")
