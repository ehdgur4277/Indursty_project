import numpy as np
from faster_whisper import WhisperModel

from config import MODEL_SIZE, DEVICE, COMPUTE_TYPE, BEAM_SIZE

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        print(f"[STT ] Whisper '{MODEL_SIZE}' 로드 중...")
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def transcribe(audio: np.ndarray) -> str:
    """Whisper로 음성을 텍스트로 변환."""
    model = _get_model()
    print("[STT ] 인식 중...")
    segments, _ = model.transcribe(
        audio,
        language="ko",
        beam_size=BEAM_SIZE,
        vad_filter=True,
    )
    return " ".join(seg.text for seg in segments).strip()
