"""
KWS 단독 테스트 스크립트

사용법:
    python3 kws_test.py
    python3 kws_test.py --audio /path/to/file.m4a
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_ROOT, "Whisper"))
sys.path.insert(0, os.path.join(_ROOT, "KWS"))

from audio_loader import load_audio
from kws import detect_wake_word
from config import AUDIO_PATH, WAKE_WORD

DATA_DIR = os.path.join(_ROOT, "data")
TEST_FILES = [f for f in os.listdir(DATA_DIR) if f.endswith((".m4a", ".wav"))]


def test_file(audio_path: str) -> None:
    fname = os.path.basename(audio_path)
    print(f"\n{'=' * 40}")
    print(f"파일: {fname}")
    audio = load_audio(audio_path)
    result = detect_wake_word(audio)
    print(f"결과: {'✓ 웨이크 워드 감지됨' if result else '✗ 웨이크 워드 없음'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=None, help="단일 파일 경로 (없으면 data/ 전체 테스트)")
    args = parser.parse_args()

    print(f"\n[KWS 테스트] 웨이크 워드: '{WAKE_WORD}'")

    if args.audio:
        test_file(args.audio)
    else:
        print(f"data/ 폴더의 파일 {len(TEST_FILES)}개 테스트")
        for fname in sorted(TEST_FILES):
            test_file(os.path.join(DATA_DIR, fname))

    print(f"\n{'=' * 40}\n")


if __name__ == "__main__":
    main()
