"""
TTS 데이터 생성 스크립트

명령어 14개를 gTTS로 합성 → 16kHz mono wav 저장 → ground_truth_tts.csv 생성

사용법:
    python3 generate_tts.py
    python3 generate_tts.py --out_dir ../data/tts --csv ground_truth_tts.csv
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

try:
    from gtts import gTTS
except ImportError:
    print("gTTS 미설치: pip install gtts")
    sys.exit(1)

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_ROOT, "Whisper"))
from command_parser import COMMAND_TO_TASK


def _to_filename(cmd: str) -> str:
    return "tts_" + cmd.replace(" ", "_") + ".wav"


def generate(out_dir: str, csv_path: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for cmd, task_id in COMMAND_TO_TASK.items():
        wav_path = os.path.join(out_dir, _to_filename(cmd))

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3 = tmp.name

        try:
            gTTS(cmd, lang="ko").save(tmp_mp3)
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3,
                 "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[생성] {_to_filename(cmd)}  →  {task_id}")
        finally:
            os.unlink(tmp_mp3)

        rows.append({
            "audio_path":           os.path.abspath(wav_path),
            "ground_truth_text":    cmd,
            "ground_truth_task_id": task_id,
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_path", "ground_truth_text", "ground_truth_task_id"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n총 {len(rows)}개 생성 완료")
    print(f"CSV  : {csv_path}")
    print(f"WAV  : {out_dir}/")


def main():
    _eval_dir = os.path.dirname(__file__)
    default_out = os.path.abspath(os.path.join(_eval_dir, "..", "data", "tts"))
    default_csv = os.path.join(_eval_dir, "ground_truth_tts.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=default_out, help="wav 저장 디렉토리")
    parser.add_argument("--csv",     default=default_csv, help="출력 CSV 경로")
    args = parser.parse_args()

    generate(args.out_dir, args.csv)


if __name__ == "__main__":
    main()
