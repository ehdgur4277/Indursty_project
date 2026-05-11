import numpy as np
from faster_whisper import WhisperModel

from config import MODEL_SIZE, DEVICE, COMPUTE_TYPE, BEAM_SIZE


def transcribe(audio: np.ndarray) -> str:
    """Whisper로 음성을 텍스트로 변환."""
    print(f"[STT ] Whisper '{MODEL_SIZE}' 로드 중...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    print("[STT ] 인식 중...")
    segments, _ = model.transcribe(
        audio,
        language="ko",
        beam_size=BEAM_SIZE,
        vad_filter=False,
    )
    return " ".join(seg.text for seg in segments).strip()
