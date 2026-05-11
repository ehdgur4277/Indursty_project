from __future__ import annotations

import numpy as np
import torch
import torchaudio
from df.enhance import enhance, init_df

_model = None
_df_state = None
_DFN_SR = None
_PIPE_SR = 16000


def _get_model():
    global _model, _df_state, _DFN_SR
    if _model is None:
        _model, _df_state, _ = init_df(
            default_model="DeepFilterNet3",
            log_level="WARNING",
            log_file=None,
        )
        _DFN_SR = _df_state.sr()  # 48000
        _model.eval()
    return _model, _df_state


def denoise(audio: np.ndarray, sr: int = _PIPE_SR) -> np.ndarray:
    """노이즈 제거. 입출력: float32 numpy, mono, sr Hz."""
    model, df_state = _get_model()
    t = torch.from_numpy(audio).unsqueeze(0)                       # [1, T]
    up = torchaudio.functional.resample(t, sr, _DFN_SR)           # [1, T_48k]
    enhanced = enhance(model, df_state, up)                         # [1, T_48k]
    down = torchaudio.functional.resample(enhanced, _DFN_SR, sr)   # [1, T]
    return down.squeeze(0).numpy().astype(np.float32)
