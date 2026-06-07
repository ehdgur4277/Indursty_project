"""
KWS (Keyword Wake Word Spotting) — openWakeWord 기반

사용법:
    from kws import detect_wake_word
    triggered = detect_wake_word(audio)   # audio: float32 np.ndarray, 16kHz mono
"""

import os
import sys
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "Whisper"))
from config import WAKE_WORD, KWS_THRESHOLD

from openwakeword.utils import AudioFeatures

_ONNX_PATH = os.path.join(os.path.dirname(__file__), "oww_model", f"{WAKE_WORD}.onnx")
_CLIP_LEN   = 32000   # 2초

_af:      AudioFeatures | None = None
_session: ort.InferenceSession | None = None


def _get_model():
    global _af, _session
    if _af is None:
        _af = AudioFeatures()
    if _session is None:
        if not os.path.exists(_ONNX_PATH):
            raise FileNotFoundError(
                f"ONNX 모델 없음: {_ONNX_PATH}\n"
                "KWS/train_oww.py를 먼저 실행하세요."
            )
        _session = ort.InferenceSession(_ONNX_PATH, providers=["CPUExecutionProvider"])
    return _af, _session


def detect_wake_word(audio: np.ndarray, threshold: float = KWS_THRESHOLD) -> bool:
    """
    오디오에서 웨이크 워드 감지.

    Args:
        audio: float32 np.ndarray, mono, 16kHz
        threshold: 감지 임계값 (0~1, 높을수록 엄격)

    Returns:
        True if wake word detected
    """
    af, session = _get_model()

    # 클립 길이 맞추기 (패딩 or 크롭)
    if len(audio) < _CLIP_LEN:
        audio = np.pad(audio, (0, _CLIP_LEN - len(audio)))
    else:
        audio = audio[:_CLIP_LEN]

    # float32 → int16
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    batch = audio_int16[np.newaxis, :]   # (1, CLIP_LEN)

    emb   = np.array(af.embed_clips(batch), dtype=np.float32)  # (1, 16, 96)
    score = session.run(None, {"input": emb})[0][0]             # scalar

    detected = float(score) >= threshold
    print(f"[KWS ] score={score:.3f}  threshold={threshold}  "
          f"{'감지' if detected else '미감지'}")
    return detected
