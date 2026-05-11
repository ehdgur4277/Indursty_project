import argparse

from config import AUDIO_PATH
from audio_loader import load_audio
from vad import extract_speech
from stt import transcribe
from command_parser import normalize_korean_command, parse_command


def run(audio_path: str) -> str:
    print(f"\n[입력] {audio_path}")

    print("[로드] 오디오 변환 중...")
    audio = load_audio(audio_path)

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
    parser.add_argument("--audio", default=AUDIO_PATH, help="오디오 파일 경로")
    args = parser.parse_args()
    run(args.audio)
