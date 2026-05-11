import subprocess
import tempfile
import os
import numpy as np

from config import SAMPLE_RATE


def load_audio(path: str) -> np.ndarray:
    """ffmpeg로 오디오 파일을 16kHz mono float32 numpy 배열로 변환."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", tmp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    raw = np.fromfile(tmp_path, dtype=np.int16)[22:]
    os.unlink(tmp_path)
    return raw.astype(np.float32) / 32768.0
