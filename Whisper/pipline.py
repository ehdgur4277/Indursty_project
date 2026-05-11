"""
파일 기반 파이프라인: .m4a → Silero VAD → Whisper STT → Command Parser → Task ID

사용법:
    python pipeline.py
    python pipeline.py --audio /path/to/file.m4a

설치:
    pip install faster-whisper torch torchaudio silero-vad ffmpeg-python
    (ffmpeg 바이너리도 필요: sudo apt install ffmpeg)
"""

import argparse
import subprocess
import tempfile
import os
import numpy as np
import torch
from faster_whisper import WhisperModel
from command_parser import normalize_korean_command, parse_command

# =============================================================================
# 설정
# =============================================================================

AUDIO_PATH   = "/data/dhpark/Indursty_project/data/Red_noisy.m4a"  # 기본 경로
MODEL_SIZE   = "medium"
DEVICE       = "cuda"
COMPUTE_TYPE = "float16"
BEAM_SIZE    = 7

SAMPLE_RATE      = 16000
VAD_THRESHOLD    = 0.5   # 소음 많으면 0.6~0.7로 올릴 것

# =============================================================================
# m4a → 16kHz mono WAV (numpy array)
# =============================================================================

def load_audio(path: str) -> np.ndarray:
    """ffmpeg로 m4a를 16kHz mono float32 numpy 배열로 변환."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "wav",
            tmp_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # WAV 헤더(44바이트) 스킵 후 float32 변환
    raw = np.fromfile(tmp_path, dtype=np.int16)[22:]  # 44바이트 / 2
    os.unlink(tmp_path)
    return raw.astype(np.float32) / 32768.0

# =============================================================================
# Silero VAD: 발화 구간만 추출
# =============================================================================

def extract_speech(audio: np.ndarray) -> np.ndarray:
    """VAD로 묵음/노이즈 제거 후 발화 구간만 이어붙여 반환."""
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        verbose=False,
    )
    get_speech_timestamps = vad_utils[0]

    tensor = torch.from_numpy(audio)
    timestamps = get_speech_timestamps(
        tensor,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
    )

    if not timestamps:
        print("[VAD] 발화 구간 없음 — 전체 오디오를 그대로 사용")
        return audio

    # 발화 구간만 이어붙임
    speech_chunks = [audio[t["start"]: t["end"]] for t in timestamps]
    speech_audio = np.concatenate(speech_chunks)

    total = len(audio) / SAMPLE_RATE
    kept  = len(speech_audio) / SAMPLE_RATE
    print(f"[VAD] 전체 {total:.1f}s → 발화 {kept:.1f}s ({len(timestamps)}개 구간)")
    return speech_audio

# =============================================================================
# 메인
# =============================================================================

def main(audio_path: str):
    print(f"\n[입력] {audio_path}")

    # 1. 오디오 로드
    print("[로드] m4a → numpy 변환 중...")
    audio = load_audio(audio_path)

    # 2. VAD
    print("[VAD ] 발화 구간 추출 중...")
    speech_audio = extract_speech(audio)

    # 3. Whisper
    print(f"[STT ] Whisper '{MODEL_SIZE}' 로드 중...")
    whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    print("[STT ] 인식 중...")
    segments, _ = whisper_model.transcribe(
        speech_audio,
        language="ko",
        beam_size=BEAM_SIZE,
        vad_filter=False,  # VAD 앞단에서 이미 처리
    )
    raw_text = " ".join(seg.text for seg in segments).strip()

    # 4. 정규화 + 파싱
    normalized = normalize_korean_command(raw_text)
    result     = parse_command(normalized)

    # 5. 출력
    print("\n" + "="*40)
    print(f"원문     : {raw_text}")
    print(f"정규화   : {normalized}")
    print(f"매핑     : {result['matched']}")
    print(f"유사도   : {result['similarity']}")
    print(f"status   : {result['status']}  ({result['reason']})")
    print(f"Task ID  : {result['task_id']}")
    print("="*40 + "\n")

    # ── VLA 연결 포인트 ───────────────────────
    # send_to_vla(result["task_id"])
    # ─────────────────────────────────────────

    return result["task_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=AUDIO_PATH, help="m4a 파일 경로")
    args = parser.parse_args()
    main(args.audio)
