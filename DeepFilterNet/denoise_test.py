"""
DeepFilterNet 노이즈 제거 실험 스크립트.

사용법:
    python3 denoise_test.py
    python3 denoise_test.py --audio /path/to/file.m4a
    python3 denoise_test.py --audio /path/to/file.m4a --out_dir ./output
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torchaudio

sys.path.insert(0, os.path.dirname(__file__))
from denoise import denoise

DEFAULT_AUDIO = "/data/dhpark/Indursty_project/data/Red_noisy.m4a"
SAMPLE_RATE   = 16000


def load_audio(path: str) -> np.ndarray:
    """m4a/wav 파일 → 16kHz mono float32 numpy."""
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(dim=0)          # stereo → mono
    return wav.numpy().astype(np.float32)


def rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio ** 2))
    return 20 * np.log10(rms + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",   default=DEFAULT_AUDIO, help="입력 오디오 파일 경로")
    parser.add_argument("--out_dir", default="./output",    help="출력 WAV 저장 폴더")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.audio))[0]

    # ── 로드 ────────────────────────────────────────────────────────────────
    print(f"\n[로드] {args.audio}")
    audio_in = load_audio(args.audio)
    print(f"      길이: {len(audio_in) / SAMPLE_RATE:.2f}s  |  RMS: {rms_db(audio_in):.1f} dBFS")

    # ── 노이즈 제거 ──────────────────────────────────────────────────────────
    print("[DFN ] DeepFilterNet3 노이즈 제거 중...")
    audio_out = denoise(audio_in, sr=SAMPLE_RATE)
    print(f"       완료  |  RMS: {rms_db(audio_out):.1f} dBFS")

    # ── 저장 ────────────────────────────────────────────────────────────────
    path_in  = os.path.join(args.out_dir, f"{base}_original.wav")
    path_out = os.path.join(args.out_dir, f"{base}_denoised.wav")

    sf.write(path_in,  audio_in,  SAMPLE_RATE)
    sf.write(path_out, audio_out, SAMPLE_RATE)

    print(f"\n[저장] 원본   → {path_in}")
    print(f"[저장] 제거후 → {path_out}")
    print("\n두 파일을 재생해 비교하세요.")


if __name__ == "__main__":
    main()
