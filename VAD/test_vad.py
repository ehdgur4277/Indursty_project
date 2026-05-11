# VAD/test_vad.py

import torch
import torchaudio
import os

# =========================
# 설정
# =========================
INPUT_PATH = "/data/dhpark/Indursty_project/data/Red_noisy.m4a"  # 테스트할 파일
OUTPUT_DIR = "VAD/output"                      # 저장 폴더
SAMPLE_RATE = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# VAD 모델 로드 (Silero - 사전학습)
# =========================
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)
(get_speech_timestamps, _, read_audio, *_) = utils

# =========================
# m4a 로드 → 16kHz mono 변환
# =========================
wav, sr = torchaudio.load(INPUT_PATH)
if sr != SAMPLE_RATE:
    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
wav = wav.mean(dim=0)  # stereo면 mono로

# =========================
# VAD 음성 구간 감지
# =========================
speech_timestamps = get_speech_timestamps(
    wav,
    vad_model,
    sampling_rate=SAMPLE_RATE,
    threshold=0.7,
    min_speech_duration_ms=500,
    min_silence_duration_ms=300,
)

print(f"감지된 음성 구간: {len(speech_timestamps)}개")

# =========================
# 구간별 .wav 저장
# =========================
input_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]

for i, ts in enumerate(speech_timestamps):
    start, end = ts['start'], ts['end']
    duration = (end - start) / SAMPLE_RATE
    print(f"  구간 {i+1}: {start/SAMPLE_RATE:.2f}s ~ {end/SAMPLE_RATE:.2f}s ({duration:.2f}s)")

    segment = wav[start:end].unsqueeze(0)
    out_path = os.path.join(OUTPUT_DIR, f"{input_name}_segment_{i+1:03d}.wav")
    torchaudio.save(out_path, segment, SAMPLE_RATE)
    print(f"  저장: {out_path}")

print("\n=== VAD 완료 ===")