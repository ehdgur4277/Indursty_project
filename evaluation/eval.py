"""
STT 정확도 평가 모듈

DFN 유/무 두 조건에서 각 오디오 파일을 실행하고
CER(문자 오류율) / 명령어 인식률을 비교 출력한다.

사용법:
    python3 eval.py
    python3 eval.py --csv ground_truth.csv
    python3 eval.py --no_dfn          # DFN 없이만 실행
"""

import argparse
import csv
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_ROOT, "Whisper"))
sys.path.insert(0, os.path.join(_ROOT, "DeepFilterNet"))

from audio_loader import load_audio
from vad import extract_speech
from stt import transcribe
from command_parser import normalize_korean_command, parse_command
from denoise import denoise

try:
    from jiwer import cer as jiwer_cer
    def calc_cer(ref: str, hyp: str) -> float:
        return jiwer_cer(ref, hyp)
except ImportError:
    # jiwer 없을 때 직접 구현 (Levenshtein 기반 CER)
    def calc_cer(ref: str, hyp: str) -> float:
        ref_c = ref.replace(" ", "")
        hyp_c = hyp.replace(" ", "")
        m, n = len(ref_c), len(hyp_c)
        if m == 0:
            return 0.0 if n == 0 else 1.0
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[:], i
            for j in range(1, n + 1):
                dp[j] = prev[j-1] if ref_c[i-1] == hyp_c[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
        return dp[n] / m


DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "ground_truth.csv")


def run_single(audio_path: str, use_dfn: bool) -> dict:
    audio = load_audio(audio_path)
    if use_dfn:
        audio = denoise(audio)
    speech = extract_speech(audio)
    raw_text = transcribe(speech)
    normalized = normalize_korean_command(raw_text)
    result = parse_command(normalized)
    return {
        "raw_text": raw_text,
        "task_id": result["task_id"],
        "matched": result["matched"],
        "similarity": result["similarity"],
        "status": result["status"],
    }


def evaluate(csv_path: str, use_dfn_list: list[bool]) -> None:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    conditions = {dfn: {"cer_list": [], "correct": 0, "total": 0} for dfn in use_dfn_list}
    results_table = []

    for row in rows:
        audio_path  = row["audio_path"]
        gt_text     = row["ground_truth_text"]
        gt_task_id  = row["ground_truth_task_id"]
        fname       = os.path.basename(audio_path)

        entry = {"file": fname, "gt_text": gt_text, "gt_task": gt_task_id}

        for dfn in use_dfn_list:
            tag = "DFN_ON" if dfn else "DFN_OFF"
            print(f"  [{tag}] {fname} ...", end=" ", flush=True)
            out = run_single(audio_path, dfn)
            print(out["raw_text"][:30])

            cer_val = calc_cer(gt_text, out["raw_text"])
            correct = (out["task_id"] == gt_task_id)

            conditions[dfn]["cer_list"].append(cer_val)
            conditions[dfn]["correct"] += int(correct)
            conditions[dfn]["total"]   += 1

            entry[f"{tag}_text"]    = out["raw_text"]
            entry[f"{tag}_task"]    = out["task_id"]
            entry[f"{tag}_cer"]     = f"{cer_val:.3f}"
            entry[f"{tag}_correct"] = "O" if correct else "X"

        results_table.append(entry)

    # ── 결과 출력 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'파일':<22} ", end="")
    for dfn in use_dfn_list:
        tag = "DFN_ON " if dfn else "DFN_OFF"
        print(f"{tag}(CER / 인식)  ", end="")
    print()
    print("-" * 72)

    for entry in results_table:
        print(f"{entry['file']:<22} ", end="")
        for dfn in use_dfn_list:
            tag = "DFN_ON" if dfn else "DFN_OFF"
            cer_s    = entry.get(f"{tag}_cer", "-")
            correct  = entry.get(f"{tag}_correct", "-")
            task     = entry.get(f"{tag}_task", "-")
            print(f"CER={cer_s}  {correct}({task:<20})  ", end="")
        print()

    print("=" * 72)
    for dfn in use_dfn_list:
        tag    = "DFN_ON " if dfn else "DFN_OFF"
        stats  = conditions[dfn]
        avg_cer = np.mean(stats["cer_list"]) if stats["cer_list"] else 0.0
        acc     = stats["correct"] / stats["total"] * 100 if stats["total"] else 0.0
        print(f"[{tag}]  평균 CER: {avg_cer:.3f}  |  명령어 인식률: {acc:.1f}%  ({stats['correct']}/{stats['total']})")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default=DEFAULT_CSV, help="ground truth CSV 경로")
    parser.add_argument("--no_dfn", action="store_true",  help="DFN 없이만 실행")
    parser.add_argument("--dfn_only", action="store_true", help="DFN 있을 때만 실행")
    args = parser.parse_args()

    if args.no_dfn:
        conditions = [False]
    elif args.dfn_only:
        conditions = [True]
    else:
        conditions = [False, True]   # 기본: 둘 다 비교

    print(f"\n[EVAL] ground truth: {args.csv}")
    print(f"[EVAL] 실행 조건: {['DFN_OFF' if not d else 'DFN_ON' for d in conditions]}\n")
    evaluate(args.csv, conditions)


if __name__ == "__main__":
    main()
