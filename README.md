# Industry Voice Command Pipeline

산업 현장 음성 명령 → 로봇 Task ID 변환 파이프라인

---

## 전체 파이프라인

```
오디오 입력 (m4a / wav)
        │
        ▼
┌─────────────────┐
│  Audio Loader   │  ffmpeg → 16kHz mono float32
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DeepFilterNet  │  신경망 기반 노이즈 제거 (DeepFilterNet3)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      VAD        │  Silero VAD — 발화 구간만 추출
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Whisper STT   │  faster-whisper medium (한국어)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Command Parser  │  Levenshtein 퍼지 매칭 → Task ID
└────────┬────────┘
         │
         ▼
     Task ID
  (→ 로봇 전달)
```

> **미구현 단계** (추후 추가 예정)
> - KWS (Keyword Wake Word Spotting)
> - 실시간 마이크 스트리밍

---

## 디렉토리 구조

```
Indursty_project/
├── pipeline/
│   └── pipeline.py        # 전체 통합 파이프라인 실행 진입점
│
├── DeepFilterNet/
│   ├── denoise.py         # DeepFilterNet3 노이즈 제거 모듈
│   └── denoise_test.py    # 노이즈 제거 단독 테스트
│
├── Whisper/
│   ├── audio_loader.py    # ffmpeg 오디오 로더
│   ├── vad.py             # Silero VAD 발화 추출
│   ├── stt.py             # faster-whisper STT
│   ├── command_parser.py  # 명령어 파싱 및 Task ID 매핑
│   ├── config.py          # 공통 설정값
│   ├── pipeline.py        # Whisper 단독 파이프라인 (DFN 미포함)
│   └── finetune/
│       ├── prepare_dataset.py          # 학습 데이터셋 생성
│       ├── train.py                    # Whisper 파인튜닝
│       └── convert_to_faster_whisper.sh
│
├── VAD/
│   └── test_vad.py        # VAD 단독 테스트
│
└── data/
    └── *.m4a              # 테스트용 녹음 파일
```

---

## 실행 방법

### 전체 파이프라인
```bash
cd pipeline
python3 pipeline.py --audio ../data/Red_noisy.m4a
```

### DeepFilterNet 단독 테스트
```bash
cd DeepFilterNet
python3 denoise_test.py --audio ../data/Red_noisy.m4a
```

### Whisper 파이프라인 (DFN 미포함)
```bash
cd Whisper
python3 pipeline.py --audio ../data/Red_noisy.m4a
```

---

## Task ID 목록

| 명령어 | Task ID |
|--------|---------|
| 빨간 통 집어 | `TASK_PICK_RED` |
| 노란 통 집어 | `TASK_PICK_YELLOW` |
| 박스 집어 | `TASK_PICK_BOX` |
| 왼쪽 빨간 통 집어 | `TASK_PICK_RED_LEFT` |
| 오른쪽 빨간 통 집어 | `TASK_PICK_RED_RIGHT` |
| 왼쪽 노란 통 집어 | `TASK_PICK_YELLOW_LEFT` |
| 오른쪽 노란 통 집어 | `TASK_PICK_YELLOW_RIGHT` |
| 빨간 통 옮겨 | `TASK_MOVE_RED` |
| 노란 통 옮겨 | `TASK_MOVE_YELLOW` |
| 박스 옮겨 | `TASK_MOVE_BOX` |

---

## 주요 설정 (`Whisper/config.py`)

| 항목 | 값 | 설명 |
|------|----|------|
| `MODEL_SIZE` | `medium` | Whisper 모델 크기 |
| `DEVICE` | `cuda` | 추론 장치 |
| `COMPUTE_TYPE` | `float16` | 연산 정밀도 |
| `SAMPLE_RATE` | `16000` | 오디오 샘플레이트 (Hz) |
| `VAD_THRESHOLD` | `0.5` | Silero VAD 발화 감도 |
