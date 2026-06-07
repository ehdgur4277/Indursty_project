"""
openWakeWord 기반 커스텀 웨이크 워드 학습 스크립트

Positive: "피식아" TTS + 속도/피치/노이즈 augmentation
Negative: 명령어 오디오 + 순수 노이즈

사용법:
    python3 train_oww.py
    python3 train_oww.py --epochs 50 --out_dir ./oww_model
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_ROOT, "Whisper"))

from openwakeword.utils import AudioFeatures
from gtts import gTTS

SAMPLE_RATE  = 16000
CLIP_LEN     = 32000          # 2초 — openWakeWord 기본 window
EMBED_SHAPE  = (16, 96)
SNR_LIST     = [5, 10, 15, 20]
SPEED_RATIOS = [0.85, 0.95, 1.0, 1.05, 1.15]

NEGATIVE_DIR  = os.path.join(_ROOT, "data", "tts")                    # 명령어 TTS
REAL_POS_DIR  = os.path.join(_ROOT, "KWS", "recordings", "pos")       # 실제 녹음 positive
REAL_NEG_DIR  = os.path.join(_ROOT, "KWS", "recordings", "neg")       # 실제 녹음 negative
CV_NEG_DIR    = os.path.join(_ROOT, "data", "common_voice_neg")        # zeroth_korean


# ── 유틸리티 ───────────────────────────────────────────────────────────────────

def _tts_wav(text: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3 = f.name
    wav = mp3.replace(".mp3", ".wav")
    try:
        gTTS(text, lang="ko").save(mp3)
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3,
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", wav],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        audio, _ = sf.read(wav)
        return audio.astype(np.float32)
    finally:
        for p in (mp3, wav):
            if os.path.exists(p): os.unlink(p)


def _change_speed(audio: np.ndarray, ratio: float) -> np.ndarray:
    """ffmpeg atempo 필터로 속도 변경."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        in_wav = f.name
    out_wav = in_wav.replace(".wav", "_out.wav")
    try:
        sf.write(in_wav, audio, SAMPLE_RATE)
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_wav,
             "-filter:a", f"atempo={ratio}", out_wav],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        result, _ = sf.read(out_wav)
        return result.astype(np.float32)
    finally:
        for p in (in_wav, out_wav):
            if os.path.exists(p): os.unlink(p)


def _add_noise(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sr  = np.sqrt(np.mean(signal**2)) + 1e-9
    nr  = np.sqrt(np.mean(noise **2)) + 1e-9
    noise = noise * sr / nr / (10 ** (snr_db / 20))
    return np.clip(signal + noise, -1.0, 1.0)


def _pink_noise(n: int) -> np.ndarray:
    cols = 16
    rows = int(np.ceil(n / cols)) + 1
    c = np.cumsum(np.random.randn(rows, cols), axis=0).flatten()[:n]
    c -= c.mean(); return (c / (c.std() + 1e-9)).astype(np.float32)


def _to_clip(audio: np.ndarray) -> np.ndarray:
    """CLIP_LEN에 맞게 패딩 or 랜덤 크롭, int16 반환."""
    if len(audio) < CLIP_LEN:
        audio = np.pad(audio, (0, CLIP_LEN - len(audio)))
    else:
        start = np.random.randint(0, len(audio) - CLIP_LEN + 1)
        audio = audio[start: start + CLIP_LEN]
    return (audio * 32767).astype(np.int16)


# ── 데이터 생성 ────────────────────────────────────────────────────────────────

def build_positive(wake_word: str) -> list[np.ndarray]:
    clips = []

    # ── TTS 합성음 ──────────────────────────────────────────────
    print(f"[POS] '{wake_word}' TTS + augmentation 생성 중...")
    base = _tts_wav(wake_word)
    for ratio in SPEED_RATIOS:
        sped = _change_speed(base, ratio)
        clips.append(_to_clip(sped))
        for snr in SNR_LIST:
            for noise in [_pink_noise(len(sped)), np.random.randn(len(sped)).astype(np.float32)]:
                clips.append(_to_clip(_add_noise(sped, noise, snr)))

    # ── 실제 목소리 녹음 ────────────────────────────────────────
    if os.path.isdir(REAL_POS_DIR):
        real_files = [
            os.path.join(REAL_POS_DIR, f)
            for f in sorted(os.listdir(REAL_POS_DIR))
            if f.endswith(".wav")
        ]
        print(f"[POS] 실제 녹음 {len(real_files)}개 + augmentation 추가 중...")
        for path in real_files:
            audio, _ = sf.read(path)
            audio = audio.astype(np.float32)
            # clean
            clips.append(_to_clip(audio))
            # speed augmentation
            for ratio in [0.9, 1.0, 1.1]:
                sped = _change_speed(audio, ratio)
                clips.append(_to_clip(sped))
                # noisy
                for snr in SNR_LIST:
                    noise = _pink_noise(len(sped))
                    clips.append(_to_clip(_add_noise(sped, noise, snr)))

    print(f"  → Positive 총 {len(clips)}개")
    return clips


def build_negative() -> list[np.ndarray]:
    clips = []

    # ── TTS 명령어 ──────────────────────────────────────────────
    print("[NEG] 명령어 TTS + 노이즈 로드 중...")
    tts_files = [
        os.path.join(NEGATIVE_DIR, f)
        for f in os.listdir(NEGATIVE_DIR) if f.endswith(".wav")
    ]
    for path in tts_files:
        audio, _ = sf.read(path)
        audio = audio.astype(np.float32)
        clips.append(_to_clip(audio))
        for snr in SNR_LIST:
            noise = _pink_noise(len(audio))
            clips.append(_to_clip(_add_noise(audio, noise, snr)))

    # ── 실제 녹음 neg_* ─────────────────────────────────────────
    if os.path.isdir(REAL_NEG_DIR):
        real_files = [
            os.path.join(REAL_NEG_DIR, f)
            for f in sorted(os.listdir(REAL_NEG_DIR))
            if f.endswith(".wav")
        ]
        print(f"[NEG] 실제 녹음 {len(real_files)}개 + augmentation 추가 중...")
        for path in real_files:
            audio, _ = sf.read(path)
            audio = audio.astype(np.float32)
            clips.append(_to_clip(audio))
            for snr in SNR_LIST:
                noise = _pink_noise(len(audio))
                clips.append(_to_clip(_add_noise(audio, noise, snr)))

    # ── zeroth_korean 일반 발화 ──────────────────────────────────
    if os.path.isdir(CV_NEG_DIR):
        cv_files = [
            os.path.join(CV_NEG_DIR, f)
            for f in sorted(os.listdir(CV_NEG_DIR)) if f.endswith(".wav")
        ]
        print(f"[NEG] zeroth_korean {len(cv_files)}개 추가 중...")
        for path in cv_files:
            audio, _ = sf.read(path)
            audio = audio.astype(np.float32)
            clips.append(_to_clip(audio))
            # 일부만 노이즈 추가 (너무 많아지지 않게)
            for snr in [10, 20]:
                noise = _pink_noise(len(audio))
                clips.append(_to_clip(_add_noise(audio, noise, snr)))

    # ── 순수 노이즈 (FP 억제용) ─────────────────────────────────
    for _ in range(len(tts_files) * 2):
        noise = _pink_noise(CLIP_LEN)
        clips.append((noise * 32767).astype(np.int16))

    print(f"  → Negative 총 {len(clips)}개")
    return clips


# ── 임베딩 추출 ────────────────────────────────────────────────────────────────

def extract_embeddings(af: AudioFeatures, clips: list[np.ndarray]) -> np.ndarray:
    batch = np.stack(clips)   # (N, CLIP_LEN)
    embs  = af.embed_clips(batch)
    return np.array(embs, dtype=np.float32)  # (N, 16, 96)


# ── 모델 정의 ──────────────────────────────────────────────────────────────────

class WakeWordDNN(nn.Module):
    def __init__(self, input_dim=16*96, hidden=128, n_blocks=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
            for _ in range(n_blocks)
        ])
        self.out = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.input_layer(self.flatten(x))
        for block in self.blocks:
            x = block(x)
        return self.out(x).squeeze(-1)


# ── 학습 ───────────────────────────────────────────────────────────────────────

def train(pos_emb: np.ndarray, neg_emb: np.ndarray,
          epochs: int, lr: float, out_dir: str) -> str:
    X = np.concatenate([pos_emb, neg_emb], axis=0)
    y = np.array([1.0]*len(pos_emb) + [0.0]*len(neg_emb), dtype=np.float32)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    dataset  = TensorDataset(X_t, y_t)
    n_val    = max(1, int(len(dataset) * 0.15))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = WakeWordDNN().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_val_loss = float("inf")
    best_state    = None

    print(f"\n[학습] device={device}  train={n_train}  val={n_val}")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
                correct += ((pred > 0.5) == yb.bool()).sum().item()
                total   += len(yb)

        val_loss = np.mean(val_losses)
        val_acc  = correct / total * 100

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs}  val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%")

    model.load_state_dict(best_state)
    return model, device


# ── ONNX 저장 ──────────────────────────────────────────────────────────────────

def save_onnx(model: nn.Module, device, out_path: str) -> None:
    model.eval()
    dummy = torch.zeros(1, *EMBED_SHAPE).to(device)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )
    print(f"[저장] {out_path}")


# ── 메인 ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wake_word", default="피식아")
    parser.add_argument("--epochs",   type=int,   default=60)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--out_dir",  default=os.path.join(os.path.dirname(__file__), "oww_model"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pos_clips = build_positive(args.wake_word)
    neg_clips = build_negative()

    print("\n[특징 추출] AudioFeatures 임베딩 중...")
    af = AudioFeatures()
    pos_emb = extract_embeddings(af, pos_clips)
    neg_emb = extract_embeddings(af, neg_clips)
    print(f"  pos_emb: {pos_emb.shape}  neg_emb: {neg_emb.shape}")

    model, device = train(pos_emb, neg_emb, args.epochs, args.lr, args.out_dir)

    onnx_path = os.path.join(args.out_dir, f"{args.wake_word}.onnx")
    save_onnx(model, device, onnx_path)
    print(f"\n완료. 모델: {onnx_path}")


if __name__ == "__main__":
    main()
