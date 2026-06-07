"""
전체 파이프라인: 오디오 파일 → VAD → KWS → Whisper STT → Task ID

사용법:
    python3 pipeline.py
    python3 pipeline.py --audio /path/to/file.m4a
    python3 pipeline.py --skip_kws   # KWS 단계 건너뜀 (평가용)
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_ROOT, "Whisper"))
sys.path.insert(0, os.path.join(_ROOT, "KWS"))

from config import AUDIO_PATH
from audio_loader import load_audio
from kws import detect_wake_word
from vad import extract_speech
from stt import transcribe
from command_parser import normalize_korean_command, parse_command


def run(audio_path: str, skip_kws: bool = False) -> str:
    print(f"\n[입력] {audio_path}")

    print("[로드] 오디오 변환 중...")
    audio = load_audio(audio_path)

    if not skip_kws:
        print("[KWS ] 웨이크 워드 감지 중...")
        if not detect_wake_word(audio):
            print("[KWS ] 웨이크 워드 미감지 — 파이프라인 종료\n")
            return "WAKE_WORD_NOT_DETECTED"

    print("[VAD ] 발화 구간 추출 중...")
    speech_audio = extract_speech(audio)

    raw_text   = transcribe(speech_audio)
    normalized = normalize_korean_command(raw_text)
    result     = parse_command(normalized)

    print("\n" + "=" * 40)
    print(f"원문     : {raw_text}")
    print(f"정규화   : {normalized}")
    print(f"매핑     : {result['matched']}")
    print(f"유사도   : {result['similarity']}")
    print(f"status   : {result['status']}  ({result['reason']})")
    print(f"Task ID  : {result['task_id']}")
    print("=" * 40 + "\n")

    # send_to_vla(result["task_id"])
    return result["task_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",    default=AUDIO_PATH, help="오디오 파일 경로")
    parser.add_argument("--skip_kws", action="store_true", help="KWS 단계 건너뜀 (평가용)")
    args = parser.parse_args()
    run(args.audio, skip_kws=args.skip_kws)
