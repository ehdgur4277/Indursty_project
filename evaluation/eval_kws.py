"""
KWS 성능 평가 스크립트

Positive: "피식아 + 명령어" 합성 오디오 → KWS가 감지해야 함  (TPR)
Negative: 명령어만 있는 오디오      → KWS가 무시해야 함  (TNR)

사용법:
    python3 eval_kws.py               # 생성 + 평가 한번에
    python3 eval_kws.py --eval_only   # 이미 생성된 파일로 평가만
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

_EVAL_DIR = os.path.dirname(__file__)
_ROOT     = os.path.dirname(_EVAL_DIR)
sys.path.insert(0, os.path.join(_ROOT, "Whisper"))
sys.path.insert(0, os.path.join(_ROOT, "KWS"))

from audio_loader import load_audio
from kws import detect_wake_word

try:
    from gtts import gTTS
except ImportError:
    print("gTTS 미설치: pip install gtts"); sys.exit(1)

TTS_CSV    = os.path.join(_EVAL_DIR, "ground_truth_tts.csv")
SAMPLE_RATE = 16000
SILENCE_SEC = 0.3   # 웨이크 워드와 명령어 사이 묵음 (초)

SNR_LIST    = [5, 10, 20]
NOISE_TYPES = ["white", "pink"]


# ── 노이즈 생성 ────────────────────────────────────────────────────────────────

def _white(n):  return np.random.randn(n).astype(np.float32)
def _pink(n):
    cols = 16
    rows = int(np.ceil(n / cols)) + 1
    c = np.cumsum(np.random.randn(rows, cols), axis=0).flatten()[:n]
    c -= c.mean(); return (c / (c.std() + 1e-9)).astype(np.float32)

_NOISE_FN = {"white": _white, "pink": _pink}

def _add_noise(signal, noise_type, snr_db):
    noise = _NOISE_FN[noise_type](len(signal))
    sr  = np.sqrt(np.mean(signal**2)) + 1e-9
    nr  = np.sqrt(np.mean(noise **2)) + 1e-9
    noise *= sr / nr / (10 ** (snr_db / 20))
    return np.clip(signal + noise, -1.0, 1.0)


# ── TTS → wav ─────────────────────────────────────────────────────────────────

def _tts_to_wav(text: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_mp3 = tmp.name
    wav_tmp = tmp_mp3.replace(".mp3", ".wav")
    try:
        gTTS(text, lang="ko").save(tmp_mp3)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3,
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", wav_tmp],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        audio, _ = sf.read(wav_tmp)
        return audio.astype(np.float32)
    finally:
        for p in (tmp_mp3, wav_tmp):
            if os.path.exists(p): os.unlink(p)


# ── 데이터 생성 ────────────────────────────────────────────────────────────────

def build_dataset(out_dir: str) -> tuple[list[dict], list[dict]]:
    """
    Returns:
        pos_rows: 웨이크 워드 있는 파일 목록 (label=1)
        neg_rows: 웨이크 워드 없는 파일 목록 (label=0)
    """
    pos_dir = os.path.join(out_dir, "positive")
    neg_dir = os.path.join(out_dir, "negative")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    with open(TTS_CSV, newline="", encoding="utf-8") as f:
        tts_rows = list(csv.DictReader(f))

    print("[생성] 웨이크 워드 TTS 생성 중...")
    wake_audio = _tts_to_wav("피식아")
    silence    = np.zeros(int(SAMPLE_RATE * SILENCE_SEC), dtype=np.float32)

    pos_rows, neg_rows = [], []

    for row in tts_rows:
        cmd_audio, _ = sf.read(row["audio_path"])
        cmd_audio = cmd_audio.astype(np.float32)
        base_name = os.path.splitext(os.path.basename(row["audio_path"]))[0]
        base_name = base_name.replace("tts_", "")

        # ── Positive (웨이크 워드 + 명령어) ──────────────────────────────────
        combined = np.concatenate([wake_audio, silence, cmd_audio])

        # clean
        p = os.path.join(pos_dir, f"pos_clean_{base_name}.wav")
        sf.write(p, combined, SAMPLE_RATE)
        pos_rows.append({"path": os.path.abspath(p), "label": 1,
                         "tag": "clean", "cmd": row["ground_truth_text"]})

        # noisy
        for nt in NOISE_TYPES:
            for snr in SNR_LIST:
                noisy = _add_noise(combined, nt, snr)
                p = os.path.join(pos_dir, f"pos_{nt}snr{snr}_{base_name}.wav")
                sf.write(p, noisy, SAMPLE_RATE)
                pos_rows.append({"path": os.path.abspath(p), "label": 1,
                                 "tag": f"{nt}_snr{snr}", "cmd": row["ground_truth_text"]})

        # ── Negative (명령어만) ───────────────────────────────────────────────
        # clean
        p = os.path.join(neg_dir, f"neg_clean_{base_name}.wav")
        sf.write(p, cmd_audio, SAMPLE_RATE)
        neg_rows.append({"path": os.path.abspath(p), "label": 0,
                         "tag": "clean", "cmd": row["ground_truth_text"]})

        # noisy
        for nt in NOISE_TYPES:
            for snr in SNR_LIST:
                noisy = _add_noise(cmd_audio, nt, snr)
                p = os.path.join(neg_dir, f"neg_{nt}snr{snr}_{base_name}.wav")
                sf.write(p, noisy, SAMPLE_RATE)
                neg_rows.append({"path": os.path.abspath(p), "label": 0,
                                 "tag": f"{nt}_snr{snr}", "cmd": row["ground_truth_text"]})

    print(f"  Positive {len(pos_rows)}개 / Negative {len(neg_rows)}개 생성 완료\n")
    return pos_rows, neg_rows


# ── 평가 ───────────────────────────────────────────────────────────────────────

def evaluate(pos_rows: list[dict], neg_rows: list[dict]) -> None:
    tags = ["clean"] + [f"{nt}_snr{snr}" for nt in NOISE_TYPES for snr in SNR_LIST]
    stats = {t: {"tp": 0, "fn": 0, "tn": 0, "fp": 0} for t in tags}

    all_rows = pos_rows + neg_rows
    total = len(all_rows)

    for i, row in enumerate(all_rows):
        print(f"\r  평가 중... {i+1}/{total}", end="", flush=True)
        audio   = load_audio(row["path"])
        detected = detect_wake_word(audio)
        tag = row["tag"]

        if row["label"] == 1:
            if detected: stats[tag]["tp"] += 1
            else:        stats[tag]["fn"] += 1
        else:
            if detected: stats[tag]["fp"] += 1
            else:        stats[tag]["tn"] += 1

    print("\n")
    print(f"{'조건':<20} {'TPR':>7} {'TNR':>7} {'FPR':>7}  TP  FN  TN  FP")
    print("-" * 65)
    for tag in tags:
        s = stats[tag]
        tp, fn, tn, fp = s["tp"], s["fn"], s["tn"], s["fp"]
        tpr = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
        fpr = fp / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
        print(f"{tag:<20} {tpr:>6.1f}% {tnr:>6.1f}% {fpr:>6.1f}%  {tp:>2}  {fn:>2}  {tn:>2}  {fp:>2}")
    print("-" * 65)


# ── 메인 ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--out_dir", default=os.path.join(_ROOT, "data", "kws_eval"))
    args = parser.parse_args()

    csv_cache = os.path.join(args.out_dir, "_rows.npz")

    if not args.eval_only:
        pos_rows, neg_rows = build_dataset(args.out_dir)
        np.savez(csv_cache, pos=str(pos_rows), neg=str(neg_rows))
    else:
        if not os.path.exists(csv_cache):
            print("캐시 없음. --eval_only 없이 먼저 실행하세요.")
            sys.exit(1)
        import ast
        data = np.load(csv_cache, allow_pickle=True)
        pos_rows = ast.literal_eval(str(data["pos"]))
        neg_rows = ast.literal_eval(str(data["neg"]))

    evaluate(pos_rows, neg_rows)


if __name__ == "__main__":
    main()
