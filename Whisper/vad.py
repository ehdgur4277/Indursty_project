import numpy as np
import torch

from config import SAMPLE_RATE, VAD_THRESHOLD


def extract_speech(audio: np.ndarray) -> np.ndarray:
    """Silero VAD로 묵음/노이즈 제거 후 발화 구간만 이어붙여 반환."""
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
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

    speech_chunks = [audio[t["start"]: t["end"]] for t in timestamps]
    speech_audio = np.concatenate(speech_chunks)

    total = len(audio) / SAMPLE_RATE
    kept  = len(speech_audio) / SAMPLE_RATE
    print(f"[VAD] 전체 {total:.1f}s → 발화 {kept:.1f}s ({len(timestamps)}개 구간)")
    return speech_audio
