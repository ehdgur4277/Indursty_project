"""
실시간 음성 명령 파이프라인

흐름:
  마이크(또는 파일) → KWS 슬라이딩 윈도우 → 웨이크워드("피식아") 감지
                    → 발화 녹음 (VAD 스트리밍으로 종료 감지)
                    → (선택) DeepFilterNet 노이즈 제거
                    → Whisper STT → 명령어 파싱 → Task ID 출력

실행:
    # 마이크 실시간 처리
    python3 rt_pipeline.py

    # 파일 입력 (마이크 없는 환경 / 테스트)
    python3 rt_pipeline.py --file ../data/Red_noisy.m4a --skip_kws

    # 노이즈 제거 비활성화
    python3 rt_pipeline.py --no_denoise
"""

import collections
import contextlib
import io
import os
import queue
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "Whisper"))
sys.path.insert(0, os.path.join(_ROOT, "KWS"))
sys.path.insert(0, os.path.join(_ROOT, "DeepFilterNet"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SAMPLE_RATE, VAD_THRESHOLD  # noqa: E402
from kws import detect_wake_word  # noqa: E402
from stt import transcribe  # noqa: E402
from denoise import denoise  # noqa: E402
from command_parser import normalize_korean_command, parse_command  # noqa: E402
from audio_stream import CHUNK_SIZE, AudioStream, FileStream, NetworkStream  # noqa: E402

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
_KWS_WINDOW  = int(2.0 * SAMPLE_RATE)                          # KWS 2초 슬라이딩 윈도우
_KWS_STRIDE  = max(1, int(0.5 * SAMPLE_RATE) // CHUNK_SIZE)    # 0.5초마다 KWS 검사
_SILENCE_END = int(1.5 * SAMPLE_RATE)   # 1.5초 침묵이면 발화 종료
_MIN_SPEECH  = int(0.4 * SAMPLE_RATE)   # 발화 최소 길이 (미만이면 무시)
_MAX_LISTEN  = int(8.0 * SAMPLE_RATE)   # 웨이크워드 후 최대 녹음 시간


def _load_silero_vad():
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        verbose=False,
    )
    return model


class RealtimePipeline:
    """
    IDLE → LISTENING → PROCESSING 상태 머신.

    Args:
        use_denoise: True면 STT 전에 DeepFilterNet3으로 노이즈 제거
    """

    def __init__(self, use_denoise: bool = True):
        print("[초기화] Silero VAD 모델 로딩...")
        self.vad_model   = _load_silero_vad()
        self.use_denoise = use_denoise
        print("[초기화] 완료. 파이프라인 준비됨\n")

    # ── 청크 단위 VAD 확률 ───────────────────────
    def _vad_prob(self, chunk: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.vad_model(torch.from_numpy(chunk), SAMPLE_RATE).item())

    # ── KWS 검사 ──
    @staticmethod
    def _check_kws(window: np.ndarray) -> bool:
        return detect_wake_word(window)

    # ── LISTENING: 발화 끝날 때까지 청크 수집 ───────
    def _listen(self, stream) -> "np.ndarray | None":
        """
        발화 구간을 수집해 numpy 배열로 반환.
        발화를 감지하지 못하거나 너무 짧으면 None 반환.
        """
        self.vad_model.reset_states()

        speech_chunks   = []
        silence_samples = 0
        total_samples   = 0
        speech_started  = False

        while total_samples < _MAX_LISTEN:
            try:
                chunk = stream.read(timeout=1.5)
            except queue.Empty:
                break

            speech_chunks.append(chunk)
            total_samples += len(chunk)

            prob = self._vad_prob(chunk)
            if prob >= VAD_THRESHOLD:
                speech_started  = True
                silence_samples = 0
            else:
                silence_samples += len(chunk)
                if speech_started and silence_samples >= _SILENCE_END:
                    break

        if not speech_started or not speech_chunks:
            return None

        audio = np.concatenate(speech_chunks)
        keep  = max(len(audio) - silence_samples, _MIN_SPEECH)
        audio = audio[:keep]

        return audio if len(audio) >= _MIN_SPEECH else None

    # ── PROCESSING: 노이즈 제거 → STT → 파싱 ──────
    def _process(self, audio: np.ndarray, stream=None):
        dur = len(audio) / SAMPLE_RATE
        print(f"[처리 ] 발화 {dur:.1f}s 처리 중...")

        if self.use_denoise:
            try:
                audio = denoise(audio)
            except Exception as e:
                print(f"[경고 ] 노이즈 제거 실패 ({e}), 원본 사용")

        raw_text   = transcribe(audio)
        normalized = normalize_korean_command(raw_text)
        result     = parse_command(normalized)

        lines = [
            "─" * 40,
            f"  원문   : {raw_text}",
            f"  매핑   : {result['matched']}",
            f"  상태   : {result['status']}",
            f"  Task   : {result['task_id']}",
            "─" * 40,
        ]
        output = "\n".join(lines)
        print(output)

        # 노트북으로도 결과 전송
        if stream is not None and hasattr(stream, "send"):
            stream.send(output)

        return result["task_id"]

    # ── 메인 루프 ────────────────────────────────
    def run(self, stream, skip_kws: bool = False):
        stream.start()

        try:
            if skip_kws:
                # ─── KWS 없이 바로 LISTENING ─────────────────────────────────
                print("[ 모드 ] KWS 건너뜀 — 바로 발화 수집 시작\n")
                audio = self._listen(stream)
                if audio is not None:
                    self._process(audio, stream)
                else:
                    print("[  -  ] 발화를 감지하지 못했습니다.\n")

            else:
                # ─── IDLE: KWS 슬라이딩 윈도우 ──────────────────────────────
                ring            = collections.deque(maxlen=_KWS_WINDOW)
                chunks_elapsed  = 0
                kws_hit_count   = 0   # 연속 감지 횟수
                print("[ 대기 ] 웨이크워드 '피식아'를 말해주세요...\n")

                while True:
                    try:
                        chunk = stream.read(timeout=30.0)
                    except queue.Empty:
                        print("[종료 ] 스트림 끝")
                        break

                    ring.extend(chunk)
                    chunks_elapsed += 1

                    # VAD 게이트: 묵음이면 KWS 실행 안 함
                    if self._vad_prob(chunk) < VAD_THRESHOLD:
                        kws_hit_count = 0
                        continue

                    if chunks_elapsed < _KWS_STRIDE:
                        continue
                    chunks_elapsed = 0

                    if len(ring) < _KWS_WINDOW:
                        continue

                    if self._check_kws(np.array(ring, dtype=np.float32)):
                        kws_hit_count += 1
                    else:
                        kws_hit_count = 0

                    # 2번 연속 감지돼야 진짜 웨이크워드로 인정
                    if kws_hit_count < 2:
                        continue
                    kws_hit_count = 0

                    # ─── LISTENING ───────────────────────────────────────────
                    print("[웨이크] '피식아' 감지! 명령을 말씀하세요...")
                    audio = self._listen(stream)

                    if audio is None:
                        print("[  -  ] 발화 없음 또는 너무 짧음, 다시 대기합니다.\n")
                        ring.clear()
                        continue

                    # ─── PROCESSING ──────────────────────────────────────────
                    self._process(audio, stream)
                    ring.clear()
                    print("[ 대기 ] 웨이크워드 '피식아'를 말해주세요...\n")

        except KeyboardInterrupt:
            print("\n[종료] 파이프라인 중단")
        finally:
            stream.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="실시간 음성 명령 파이프라인",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "예시:\n"
            "  마이크 실시간:  python3 rt_pipeline.py\n"
            "  파일 테스트:    python3 rt_pipeline.py --file ../data/Red_noisy.m4a --skip_kws\n"
            "  노이즈 제거 끔: python3 rt_pipeline.py --no_denoise"
        ),
    )
    parser.add_argument("--file",       default=None,        help="마이크 대신 사용할 오디오 파일 경로 (.m4a / .wav)")
    parser.add_argument("--network",    action="store_true",  help="노트북 마이크를 TCP로 수신 (mic_client.py와 쌍)")
    parser.add_argument("--port",       type=int, default=9876, help="--network 모드 포트 (기본 9876)")
    parser.add_argument("--skip_kws",   action="store_true",  help="웨이크워드 감지 건너뜀 (파일 테스트에 유용)")
    parser.add_argument("--no_denoise", action="store_true",  help="DeepFilterNet 노이즈 제거 비활성화")
    args = parser.parse_args()

    pipeline = RealtimePipeline(use_denoise=not args.no_denoise)

    if args.network:
        stream = NetworkStream(port=args.port)
    elif args.file:
        print(f"[ 파일 ] {args.file}\n")
        stream = FileStream(args.file)
    else:
        stream = AudioStream()

    pipeline.run(stream, skip_kws=args.skip_kws)
