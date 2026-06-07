"""
노이즈 합성 스크립트

TTS 클린 음성에 화이트/핑크 노이즈를 SNR별로 섞어 저장 + CSV 생성

사용법:
    python3 add_noise.py
    python3 add_noise.py --snr 5 10 20 --noise_type white pink
"""

import argparse
import csv
import os
import sys

import numpy as np
import soundfile as sf

_EVAL_DIR = os.path.dirname(__file__)
_ROOT     = os.path.dirname(_EVAL_DIR)

TTS_CSV   = os.path.join(_EVAL_DIR, "ground_truth_tts.csv")
SAMPLE_RATE = 16000


# ── 노이즈 생성 ────────────────────────────────────────────────────────────────

def _white_noise(n: int) -> np.ndarray:
    return np.random.randn(n).astype(np.float32)


def _pink_noise(n: int) -> np.ndarray:
    """Voss-McCartney 알고리즘 근사 (1/f 스펙트럼)."""
    cols = 16
    rows = int(np.ceil(n / cols)) + 1
    array = np.random.randn(rows, cols)
    cumsum = np.cumsum(array, axis=0)
    flat   = cumsum.flatten()[:n]
    flat  -= flat.mean()
    return (flat / (flat.std() + 1e-9)).astype(np.float32)


_NOISE_FN = {"white": _white_noise, "pink": _pink_noise}


# ── SNR 믹싱 ───────────────────────────────────────────────────────────────────

def mix_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """target SNR(dB)에 맞춰 노이즈 스케일 후 합산, [-1, 1] 클리핑."""
    sig_rms   = np.sqrt(np.mean(signal ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise  ** 2)) + 1e-9
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    mixed = signal + scaled_noise
    return np.clip(mixed, -1.0, 1.0)


# ── 메인 ───────────────────────────────────────────────────────────────────────

def generate(snr_list: list[float], noise_types: list[str]) -> None:
    with open(TTS_CSV, newline="", encoding="utf-8") as f:
        tts_rows = list(csv.DictReader(f))

    for noise_type in noise_types:
        noise_fn = _NOISE_FN[noise_type]

        for snr in snr_list:
            tag     = f"{noise_type}_snr{int(snr)}"
            out_dir = os.path.join(_ROOT, "data", "noisy", tag)
            os.makedirs(out_dir, exist_ok=True)

            csv_path = os.path.join(_EVAL_DIR, f"ground_truth_{tag}.csv")
            rows = []

            for row in tts_rows:
                clean, sr = sf.read(row["audio_path"])
                assert sr == SAMPLE_RATE, f"샘플레이트 불일치: {row['audio_path']}"

                noise  = noise_fn(len(clean))
                mixed  = mix_snr(clean, noise, snr)

                src_name  = os.path.basename(row["audio_path"])           # tts_빨간_통_집어.wav
                out_name  = src_name.replace("tts_", f"{tag}_")
                out_path  = os.path.join(out_dir, out_name)
                sf.write(out_path, mixed, SAMPLE_RATE)

                rows.append({
                    "audio_path":           os.path.abspath(out_path),
                    "ground_truth_text":    row["ground_truth_text"],
                    "ground_truth_task_id": row["ground_truth_task_id"],
                })
                print(f"[{tag}] {out_name}")

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["audio_path", "ground_truth_text", "ground_truth_task_id"]
                )
                writer.writeheader()
                writer.writerows(rows)

            print(f"→ CSV: {csv_path}\n")

    print("노이즈 합성 완료")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr",        nargs="+", type=float, default=[5, 10, 20],
                        help="SNR 레벨(dB) 리스트 (낮을수록 노이즈 많음)")
    parser.add_argument("--noise_type", nargs="+", default=["white", "pink"],
                        choices=["white", "pink"])
    args = parser.parse_args()
    generate(args.snr, args.noise_type)


if __name__ == "__main__":
    main()
