# Industry Voice Command Pipeline

산업 현장 음성 명령 → 로봇 Task ID 변환 파이프라인

**웨이크워드("피식아") 감지 → STT → 퍼지 매칭 → Task ID**

---

## 전체 구조

```
음성 입력
    │
    ▼
┌──────────┐
│   KWS    │  "피식아" 웨이크워드 감지 (openWakeWord ONNX)
└────┬─────┘
     │
     ▼
┌──────────┐
│   VAD    │  Silero VAD — 발화 구간 추출
└────┬─────┘
     │
     ▼
┌──────────┐
│   DFN    │  DeepFilterNet3 노이즈 제거 (선택)
└────┬─────┘
     │
     ▼
┌──────────┐
│  Whisper │  faster-whisper medium (한국어)
└────┬─────┘
     │
     ▼
┌──────────┐
│  Parser  │  Levenshtein 퍼지 매칭 → Task ID
└────┬─────┘
     │
     ▼
  Task ID → 로봇
```

---

## 디렉토리 구조

```
Indursty_project/
├── realtime/                    # 실시간 파이프라인 (마이크 / 네트워크)
│   ├── rt_pipeline.py           # 메인 실시간 파이프라인
│   ├── audio_stream.py          # AudioStream / FileStream / NetworkStream
│   └── mic_client.py            # 노트북 마이크 → 서버 TCP 스트리밍 클라이언트
│
├── pipeline/
│   └── pipeline.py              # 파일 기반 통합 파이프라인
│
├── Whisper/
│   ├── command_parser.py        # 명령어 파싱 및 Task ID 매핑
│   ├── stt.py                   # faster-whisper STT
│   ├── vad.py                   # Silero VAD
│   ├── audio_loader.py          # ffmpeg 오디오 로더
│   ├── config.py                # 공통 설정값
│   └── finetune/                # Whisper 파인튜닝 스크립트
│
├── KWS/
│   ├── kws.py                   # openWakeWord 웨이크워드 감지
│   ├── train_oww.py             # 웨이크워드 모델 학습
│   ├── record_wakeword.py       # 웨이크워드 녹음 도구
│   └── oww_model/               # 학습된 ONNX 모델
│
├── DeepFilterNet/
│   └── denoise.py               # DeepFilterNet3 노이즈 제거
│
├── evaluation/
│   ├── eval.py                  # STT 정확도 평가
│   ├── eval_kws.py              # KWS 성능 평가
│   ├── generate_tts.py          # TTS 평가 데이터 생성
│   ├── add_noise.py             # 노이즈 추가 유틸
│   └── ground_truth_*.csv       # 평가용 정답 데이터
│
├── data/                        # 테스트 오디오 파일
├── requirements.txt             # 전체 패키지 목록
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
```

### 2. PyTorch (환경에 맞게 선택)

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

### 3. 나머지 패키지

```bash
pip install -r requirements.txt
```

---

## 실행 방법

### 실시간 파이프라인 (`realtime/`)

#### A. 로컬 마이크 (마이크가 직접 연결된 경우)

```bash
cd realtime
python3 rt_pipeline.py
python3 rt_pipeline.py --no_denoise     # 노이즈 제거 생략 (빠름)
```

#### B. 파일 입력 (마이크 없는 환경 / 테스트)

```bash
cd realtime
python3 rt_pipeline.py --file ../data/Red_noisy.m4a --skip_kws
python3 rt_pipeline.py --file ../data/Red_noisy.m4a --skip_kws --no_denoise
```

#### C. 네트워크 마이크 (Docker / 원격 서버 환경)

마이크가 없는 서버(또는 Docker)에서 노트북 마이크를 TCP로 연결합니다.

**① 서버에서 (파이프라인 실행):**

```bash
cd realtime
python3 rt_pipeline.py --network
python3 rt_pipeline.py --network --port 9876 --no_denoise
```

**② 노트북에서 (마이크 스트리밍):**

```bash
# 패키지 설치 (최초 1회)
pip install -r requirements-mic-client.txt

# 마이크 장치 목록 확인
python3 mic_client.py --device

# 서버로 마이크 스트리밍
python3 mic_client.py <서버IP>
python3 mic_client.py <서버IP> --port 9876
```

> **VSCode + Remote SSH 환경:**
> `Ctrl+Shift+P` → `Terminal: Create New Integrated Terminal (Local)` 로 로컬 터미널을 열고 `mic_client.py`를 실행하세요.

> **Docker 환경:**
> SSH 포트포워딩으로 연결할 수 있습니다.
> ```bash
> # 노트북 터미널에서 (SSH 연결 유지)
> ssh -L 9876:<컨테이너IP>:9876 user@서버IP
>
> # 새 로컬 터미널에서
> python3 mic_client.py localhost
> ```

---

### 파일 기반 파이프라인 (`pipeline/`)

```bash
cd pipeline
python3 pipeline.py --audio ../data/Red_noisy.m4a
python3 pipeline.py --audio ../data/Red_noisy.m4a --skip_kws
```

---

## 명령어 목록 및 Task ID

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

> 정확한 발음이 아니어도 Levenshtein 퍼지 매칭으로 가장 가까운 명령어에 매핑됩니다. (임계값 0.5)

---

## 주요 설정 (`Whisper/config.py`)

| 항목 | 값 | 설명 |
|------|----|------|
| `MODEL_SIZE` | `medium` | Whisper 모델 크기 (`small`로 줄이면 빠름) |
| `DEVICE` | `cuda` | 추론 장치 (`cpu` 가능) |
| `COMPUTE_TYPE` | `float16` | 연산 정밀도 |
| `SAMPLE_RATE` | `16000` | 오디오 샘플레이트 (Hz) |
| `VAD_THRESHOLD` | `0.5` | Silero VAD 발화 감도 |
| `WAKE_WORD` | `피식아` | 웨이크워드 |
| `KWS_THRESHOLD` | `0.5` | 웨이크워드 감지 임계값 |

---

## 평가

```bash
cd evaluation

# STT 정확도 평가
python3 eval.py

# KWS 성능 평가
python3 eval_kws.py

# TTS 평가 데이터 생성
python3 generate_tts.py
```
