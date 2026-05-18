"""
KWS (Keyword Wake Word Spotting)

faster-whisper tiny 모델로 오디오에서 웨이크 워드("피식아") 감지.

사용법:
    from kws import detect_wake_word
    triggered = detect_wake_word(audio)   # audio: float32 np.ndarray, 16kHz mono
"""

import os
import sys
import numpy as np
from faster_whisper import WhisperModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "Whisper"))
from config import WAKE_WORD, KWS_THRESHOLD, KWS_MODEL_SIZE, DEVICE, COMPUTE_TYPE


_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(KWS_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def _similarity(a: str, b: str) -> float:
    a, b = a.replace(" ", ""), b.replace(" ", "")
    if not a and not b:
        return 1.0
    dist = _levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b))


def detect_wake_word(audio: np.ndarray) -> bool:
    """
    오디오에서 웨이크 워드 감지.

    Args:
        audio: float32 np.ndarray, mono, 16kHz

    Returns:
        True if wake word detected, False otherwise
    """
    model = _get_model()
    segments, _ = model.transcribe(audio, language="ko", beam_size=1)
    text = " ".join(seg.text for seg in segments).strip()
    print(f"[KWS ] 인식 텍스트: '{text}'")

    normalized = text.replace(" ", "")
    wake = WAKE_WORD.replace(" ", "")

    # 완전 포함 우선 확인
    if wake in normalized:
        print(f"[KWS ] 웨이크 워드 감지 (exact match)")
        return True

    # 슬라이딩 윈도우로 부분 유사도 확인 (wake word 길이 기준)
    wlen = len(wake)
    best = 0.0
    for i in range(len(normalized) - wlen + 1):
        window = normalized[i: i + wlen]
        sim = _similarity(wake, window)
        if sim > best:
            best = sim

    # 전체 텍스트와의 유사도도 비교
    full_sim = _similarity(wake, normalized)
    best = max(best, full_sim)

    detected = best >= KWS_THRESHOLD
    status = f"감지 (유사도 {best:.2f})" if detected else f"미감지 (유사도 {best:.2f})"
    print(f"[KWS ] {status}")
    return detected
