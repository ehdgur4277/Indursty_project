# Industry Voice Command Pipeline

산업 현장 음성 명령 → 로봇 Task ID 변환 파이프라인

---

## 동작 흐름

```
마이크 입력 (실시간)
    │
    ▼
┌──────────┐
│   VAD    │  Silero VAD — 묵음 구간 필터링
└────┬─────┘
     │ 발화 감지 시에만
     ▼
┌──────────┐
│   KWS    │  "피식아" 웨이크워드 감지 (openWakeWord ONNX)
└────┬─────┘
     │ 감지 시
     ▼
┌──────────┐
│   VAD    │  발화 종료 감지 (1.5초 침묵)
└────┬─────┘
     │
     ▼
┌──────────┐
│   DFN    │  DeepFilterNet3 노이즈 제거 (선택)
└────┬─────┘
     │
     ▼
┌──────────┐
│  Whisper │  faster-whisper medium (한국어 STT)
└────┬─────┘
     │
     ▼
┌──────────┐
│  Parser  │  Levenshtein 퍼지 매칭 → Task ID
└────┬─────┘
     │
     ▼
  Task ID → 로봇 전달
```

---

## 디렉토리 구조

```
Indursty_project/
├── realtime/
│   ├── rt_pipeline.py           # 실시간 파이프라인 메인
│   ├── audio_stream.py          # AudioStream / FileStream / NetworkStream
│   ├── mic_client.py            # 노트북 마이크 → 서버 TCP 스트리밍
│   └── start.sh                 # 파이프라인 + bore 한 번에 실행
│
├── pipeline/
│   └── pipeline.py              # 파일 기반 파이프라인
│
├── Whisper/
│   ├── command_parser.py        # 명령어 파싱 및 Task ID 매핑
│   ├── stt.py                   # faster-whisper STT
│   ├── vad.py                   # Silero VAD
│   ├── audio_loader.py          # ffmpeg 오디오 로더
│   └── config.py                # 공통 설정
│
├── KWS/
│   ├── kws.py                   # openWakeWord 웨이크워드 감지
│   ├── train_oww.py             # 웨이크워드 모델 학습
│   ├── record_wakeword.py       # 웨이크워드 녹음 도구
│   ├── record_noise.py          # 환경 소음 negative 샘플 녹음
│   └── oww_model/               # 학습된 ONNX 모델
│
├── DeepFilterNet/
│   └── denoise.py               # DeepFilterNet3 노이즈 제거
│
├── evaluation/
│   ├── eval.py                  # STT 정확도 평가
│   ├── eval_kws.py              # KWS 성능 평가
│   ├── generate_tts.py          # TTS 평가 데이터 생성
│   └── ground_truth_*.csv       # 평가 정답 데이터
│
├── requirements.txt
└── requirements-mic-client.txt  # mic_client 전용 (경량)
```

---

## 설치

### 1. 시스템 패키지

```bash
# Ubuntu / Debian
sudo apt install ffmpeg portaudio19-dev python3-pip

# macOS
brew install ffmpeg portaudio

# Windows
winget install ffmpeg
```

### 2. PyTorch

```bash
# GPU (CUDA 11.8)
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# GPU (CUDA 12.1)
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.1.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

### 3. 패키지

```bash
pip install -r requirements.txt
```

---

## 실행 방법

### A. 로컬 마이크 (서버에 마이크가 직접 연결된 경우)

```bash
cd realtime
python3 rt_pipeline.py
python3 rt_pipeline.py --no_denoise    # 노이즈 제거 생략 (더 빠름)
```

### B. 파일 입력 (테스트용)

```bash
cd realtime
python3 rt_pipeline.py --file ../data/Red_noisy.m4a --skip_kws --no_denoise
```

### C. 네트워크 마이크 — Docker / 원격 서버 환경 ★

마이크 없는 서버에서 **노트북 마이크를 TCP로 연결**하는 방법입니다.

#### ① 서버 (Docker 터미널)

```bash
cd /data/dhpark/Indursty_project/realtime

# bore 터널 + 파이프라인 동시 시작 (Ctrl+C로 둘 다 종료)
./start.sh

# 노이즈 제거 없이 실행 (더 빠름)
./start.sh --no_denoise
```

실행하면 아래와 같이 bore 포트 번호가 출력됩니다:
```
listening at bore.pub:XXXXX   ← 이 번호 메모
```

#### ② 노트북 (로컬 터미널)

> VSCode를 쓴다면: `Ctrl+Shift+P` → `Terminal: Create New Integrated Terminal (Local)` 로 **로컬 터미널** 열기

```bash
# 최초 1회 설치
pip install sounddevice numpy

# 마이크 스트리밍 시작 (XXXXX는 bore에서 출력된 포트 번호)
python3 mic_client.py bore.pub --port XXXXX
```

연결되면 서버에서 "피식아"를 기다리고, 명령어 처리 결과가 **노트북 터미널에도 출력**됩니다.

---

## 사용 방법

1. `./start.sh` 실행 → bore 포트 번호 확인
2. 노트북에서 `mic_client.py bore.pub --port XXXXX` 실행
3. **"피식아"** 라고 말하기 → `[웨이크] '피식아' 감지!` 출력
4. 바로 이어서 명령어 말하기
5. 결과 출력 후 다시 대기 상태 → 1번부터 반복

---

## 명령어 목록

| 명령어 | Task ID |
|--------|---------|
| 빨간색 박스 집어 | `TASK_PICK_RED_BOX` |
| 파란색 박스 집어 | `TASK_PICK_BLUE_BOX` |
| 초록색 박스 집어 | `TASK_PICK_GREEN_BOX` |
| 빨간색 박스 넣어 | `TASK_PUT_RED_BOX` |
| 파란색 박스 넣어 | `TASK_PUT_BLUE_BOX` |
| 초록색 박스 넣어 | `TASK_PUT_GREEN_BOX` |
| 빨간색 박스 집어서 넣어 | `TASK_PICK_PUT_RED_BOX` |
| 파란색 박스 집어서 넣어 | `TASK_PICK_PUT_BLUE_BOX` |
| 초록색 박스 집어서 넣어 | `TASK_PICK_PUT_GREEN_BOX` |

> 발음이 정확하지 않아도 Levenshtein 퍼지 매칭으로 가장 가까운 명령어에 매핑됩니다.

---

## 주요 설정 (`Whisper/config.py`)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_SIZE` | `medium` | Whisper 모델 (`small`로 줄이면 빠름) |
| `DEVICE` | `cuda` | 추론 장치 (`cpu` 가능) |
| `COMPUTE_TYPE` | `float16` | 연산 정밀도 (CPU는 `int8`) |
| `SAMPLE_RATE` | `16000` | 오디오 샘플레이트 (Hz) |
| `VAD_THRESHOLD` | `0.5` | Silero VAD 발화 감도 |
| `WAKE_WORD` | `피식아` | 웨이크워드 |
| `KWS_THRESHOLD` | `0.7` | 웨이크워드 감지 임계값 |

---

## KWS 오감지 해결

KWS가 말하지 않아도 계속 감지되는 경우, 현재 환경 소음을 negative 샘플로 추가 후 재학습합니다.

```bash
# 1. 노트북에서 환경 소음 녹음 (30초 × 3회, 아무 말 하지 말 것)
python3 KWS/record_noise.py

# 2. 녹음된 neg_env_*.wav 파일을 서버 KWS/recordings/ 에 복사

# 3. 재학습
cd KWS
python3 train_oww.py
```

---

## 평가

```bash
cd evaluation

python3 eval.py          # STT 정확도 평가
python3 eval_kws.py      # KWS 성능 평가
python3 generate_tts.py  # TTS 평가 데이터 생성
```
