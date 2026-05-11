"""
TTS로 학습 데이터 생성 + 실제 녹음 파일 병합 → HuggingFace Dataset 형태로 저장

설치:
    pip install TTS soundfile datasets transformers librosa audiomentations
    (Coqui TTS: pip install TTS)

사용법:
    python prepare_dataset.py
    → ./dataset/ 아래에 train/test split 저장
"""

import os
import json
import csv
import random
import numpy as np
import soundfile as sf
from pathlib import Path

# =============================================================================
# 명령어 문장 정의 (라벨 = 정답 텍스트)
# =============================================================================

COMMANDS = [
    "빨간 통 집어",
    "노란 통 집어",
    "박스 집어",
    "빨간 통 왼쪽으로 집어",
    "빨간 통 오른쪽으로 집어",
    "노란 통 왼쪽으로 집어",
    "노란 통 오른쪽으로 집어",
    "왼쪽 빨간 통 집어",
    "오른쪽 빨간 통 집어",
    "왼쪽 노란 통 집어",
    "오른쪽 노란 통 집어",
    "빨간 통 옮겨",
    "노란 통 옮겨",
    "박스 옮겨",
]

# 실제 녹음 파일과 정답 텍스트 매핑 (직접 수정)
REAL_RECORDINGS = {
    "/data/dhpark/Indursty_project/data/Red_noisy.m4a":       "빨간 통 집어",
    "/data/dhpark/Indursty_project/data/Yellow_noisy.m4a":    "노란 통 집어",
    "/data/dhpark/Indursty_project/data/voice_test.m4a":      "빨간 통 집어",
    "/data/dhpark/Indursty_project/data/voice_test_yellow.m4a": "노란 통 집어",
    "/data/dhpark/Indursty_project/data/yellow_dongmin.m4a":  "노란 통 집어",
}

SAMPLE_RATE = 16000
TTS_PER_COMMAND = 5   # 명령어당 TTS 생성 개수 (화자 변형)
OUTPUT_DIR = Path(__file__).parent / "dataset"
AUDIO_DIR  = OUTPUT_DIR / "audio"

# =============================================================================
# TTS 생성
# =============================================================================

def generate_tts_audio(texts: list[str], output_dir: Path, n_per_text: int = 5):
    """Coqui TTS로 각 문장을 n_per_text개씩 생성 (속도/피치 변형)."""
    try:
        from TTS.api import TTS
    except ImportError:
        print("[TTS] Coqui TTS 미설치 → TTS 생성 건너뜀")
        print("      pip install TTS  후 재실행하세요.")
        return []

    tts = TTS(model_name="tts_models/ko/css10/vits", progress_bar=False)
    records = []

    for text in texts:
        for i in range(n_per_text):
            filename = f"tts_{text.replace(' ', '_')}_{i}.wav"
            wav_path = output_dir / filename

            # 약간의 속도 변형 (-10% ~ +10%)
            speed = random.uniform(0.9, 1.1)
            tts.tts_to_file(text=text, file_path=str(wav_path), speed=speed)

            records.append({"audio": str(wav_path), "text": text, "source": "tts"})
            print(f"  [TTS] {filename}")

    return records

# =============================================================================
# 실제 녹음 파일 로드 (m4a → wav 변환)
# =============================================================================

def load_real_recordings(recordings: dict, output_dir: Path):
    """m4a 등을 16kHz wav로 변환해서 dataset에 포함."""
    import subprocess
    records = []

    for src_path, text in recordings.items():
        if not os.path.exists(src_path):
            print(f"  [SKIP] 파일 없음: {src_path}")
            continue

        stem = Path(src_path).stem
        wav_path = output_dir / f"real_{stem}.wav"

        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path,
             "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        records.append({"audio": str(wav_path), "text": text, "source": "real"})
        print(f"  [REAL] {wav_path.name}  →  '{text}'")

    return records

# =============================================================================
# 오디오 증강 (노이즈, 피치, 속도)
# =============================================================================

def augment_records(records: list[dict], output_dir: Path, n_aug: int = 3):
    """실제 녹음 파일만 증강해서 데이터 부족 보완."""
    try:
        from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
    except ImportError:
        print("[AUG] audiomentations 미설치 → 증강 건너뜀")
        print("      pip install audiomentations  후 재실행하세요.")
        return []

    augment = Compose([
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=0.5),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    ])

    aug_records = []
    real_records = [r for r in records if r["source"] == "real"]

    for r in real_records:
        audio, sr = sf.read(r["audio"])
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        for i in range(n_aug):
            aug_audio = augment(samples=audio.astype(np.float32), sample_rate=SAMPLE_RATE)
            stem = Path(r["audio"]).stem
            out_path = output_dir / f"{stem}_aug{i}.wav"
            sf.write(str(out_path), aug_audio, SAMPLE_RATE)
            aug_records.append({"audio": str(out_path), "text": r["text"], "source": "aug"})
            print(f"  [AUG] {out_path.name}")

    return aug_records

# =============================================================================
# CSV 저장 + train/test split
# =============================================================================

def save_csv(records: list[dict], output_dir: Path, test_ratio: float = 0.1):
    random.shuffle(records)
    split = max(1, int(len(records) * test_ratio))
    test_records  = records[:split]
    train_records = records[split:]

    for name, recs in [("train.csv", train_records), ("test.csv", test_records)]:
        path = output_dir / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio", "text", "source"])
            writer.writeheader()
            writer.writerows(recs)
        print(f"\n[저장] {path}  ({len(recs)}개)")

# =============================================================================
# 메인
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []

    print("\n=== 1. 실제 녹음 파일 변환 ===")
    real = load_real_recordings(REAL_RECORDINGS, AUDIO_DIR)
    all_records.extend(real)

    print("\n=== 2. TTS 데이터 생성 ===")
    tts = generate_tts_audio(COMMANDS, AUDIO_DIR, n_per_text=TTS_PER_COMMAND)
    all_records.extend(tts)

    print("\n=== 3. 실제 녹음 증강 ===")
    aug = augment_records(real, AUDIO_DIR, n_aug=5)
    all_records.extend(aug)

    print(f"\n총 {len(all_records)}개 샘플")
    save_csv(all_records, OUTPUT_DIR)
    print("\n[완료] dataset/train.csv, dataset/test.csv 생성됨")


if __name__ == "__main__":
    main()
