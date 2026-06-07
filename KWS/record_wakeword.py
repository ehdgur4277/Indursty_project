"""
웨이크 워드 녹음 스크립트

노트북/PC 마이크로 실행:
    pip install sounddevice soundfile
    python3 record_wakeword.py
    python3 record_wakeword.py --out_dir ./recordings --duration 2.5
"""

import argparse
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000

SCRIPT = [
    # (파일명 prefix, label, 발화 텍스트, 녹음 시간(초))
    # ── A: 웨이크 워드 단독 ───────────────────────────────────────
    ("pos_normal_1",   "pos", "피식아  (보통 속도)",   2.5),
    ("pos_normal_2",   "pos", "피식아  (보통 속도)",   2.5),
    ("pos_normal_3",   "pos", "피식아  (보통 속도)",   2.5),
    ("pos_fast",       "pos", "피식아  (빠르게)",       2.0),
    ("pos_slow",       "pos", "피식아  (천천히)",       3.0),
    ("pos_quiet",      "pos", "피식아  (작은 목소리)", 2.5),
    ("pos_loud",       "pos", "피식아  (큰 목소리)",   2.5),
    # ── B: 웨이크 워드 + 명령어 ──────────────────────────────────
    ("pos_cmd_red",    "pos", "피식아 빨간 통 집어",   3.5),
    ("pos_cmd_yellow", "pos", "피식아 노란 통 집어",   3.5),
    ("pos_cmd_move",   "pos", "피식아 빨간 통 옮겨",   3.5),
    ("pos_cmd_left",   "pos", "피식아 왼쪽 빨간 통 집어", 4.0),
    # ── C: 비슷한 발음 (미감지여야 함) ───────────────────────────
    ("neg_similar_1",  "neg", "피식야  (피식아 아님)",  2.5),
    ("neg_similar_2",  "neg", "비식아  (피식아 아님)",  2.5),
    ("neg_similar_3",  "neg", "피직아  (피식아 아님)",  2.5),
    ("neg_similar_4",  "neg", "피식    (피식아 아님)",  2.0),
    # ── D: 명령어만 (미감지여야 함) ──────────────────────────────
    ("neg_cmd_1",      "neg", "빨간 통 집어",          2.5),
    ("neg_cmd_2",      "neg", "노란 통 집어",          2.5),
    ("neg_cmd_3",      "neg", "박스 집어",             2.0),
    # ── E: 일반 대화 (미감지여야 함) ─────────────────────────────
    ("neg_talk_1",     "neg", "오늘 날씨 좋다",        2.5),
    ("neg_talk_2",     "neg", "밥 먹었어",             2.0),
    ("neg_talk_3",     "neg", "잠깐만요",              2.0),
    ("neg_silence",    "neg", "(침묵 — 아무 말 하지 마세요)", 3.0),
]


def select_device() -> int:
    devices = sd.query_devices()
    inputs  = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not inputs:
        raise RuntimeError(
            "마이크 입력 장치를 찾을 수 없습니다.\n"
            "이 스크립트는 로컬 노트북/PC에서 실행하세요.\n"
            "서버(SSH 환경)에서는 마이크가 동작하지 않습니다."
        )
    if len(inputs) == 1:
        idx, dev = inputs[0]
        print(f"  마이크 자동 선택: [{idx}] {dev['name']}\n")
        return idx
    print("  사용 가능한 마이크:")
    for idx, dev in inputs:
        print(f"    [{idx}] {dev['name']}")
    choice = int(input("  번호 선택: "))
    return choice


def record_clip(duration: float, device: int) -> np.ndarray:
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio.squeeze()


def countdown(n: int = 3) -> None:
    for i in range(n, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(1)
    print("녹음 시작!\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",  default=os.path.join(os.path.dirname(__file__), "recordings"))
    parser.add_argument("--duration", type=float, default=None, help="고정 녹음 시간 (기본: 대본별 자동)")
    args = parser.parse_args()

    device = select_device()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("  웨이크 워드 녹음 스크립트")
    print(f"  총 {len(SCRIPT)}개 발화  |  저장: {args.out_dir}")
    print("=" * 50)
    print("\n  Enter를 누르면 각 발화를 시작합니다.\n")

    for i, (fname, label, text, dur) in enumerate(SCRIPT, 1):
        dur = args.duration or dur
        print(f"[{i:2d}/{len(SCRIPT)}]  {text}")
        print(f"         녹음 시간: {dur}초  |  준비되면 Enter ▶", end=" ")
        input()

        countdown(3)
        audio = record_clip(dur, device)

        out_path = os.path.join(args.out_dir, f"{fname}.wav")
        sf.write(out_path, audio, SAMPLE_RATE)
        print(f"  ✓  저장: {fname}.wav\n")

    print("=" * 50)
    print(f"  녹음 완료! 파일 {len(SCRIPT)}개 → {args.out_dir}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
