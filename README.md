# Industry Voice Command Pipeline

음성 명령으로 로봇 Task ID를 전달하는 실시간 파이프라인

**"피식아" → 명령어 인식 → Task ID → 로봇**

---

## 목차

1. [동작 흐름](#동작-흐름)
2. [디렉토리 구조](#디렉토리-구조)
3. [서버 환경 설치](#서버-환경-설치)
4. [실행 방법](#실행-방법)
5. [노트북 연결 방법](#노트북-연결-방법)
6. [명령어 목록](#명령어-목록)
7. [주요 설정](#주요-설정)
8. [평가](#평가)
9. [문제 해결](#문제-해결)

---

## 동작 흐름

```
노트북 마이크 입력
      │
      ▼  (bore TCP 터널)
  [서버 / Docker]
      │
      ▼
  VAD (Silero VAD)        ← 묵음 구간 필터링, 발화 시작 감지
      │ 발화 감지 시
      ▼
  Whisper STT ①           ← "피식아"인지 확인
      │ 웨이크워드 확인 시
      ▼
  VAD                     ← 명령어 발화 감지
      │
      ▼
  Whisper STT ②           ← 명령어 인식
      │
      ▼
  Levenshtein 퍼지 매칭   ← 인식 결과 → 가장 가까운 명령어
      │
      ▼
  Task ID 출력             ← 서버 + 노트북 동시 출력
```

---

## 디렉토리 구조

```
Indursty_project/
│
├── realtime/                      # 실시간 파이프라인 (핵심)
│   ├── rt_pipeline.py             # 메인 파이프라인
│   ├── audio_stream.py            # 오디오 스트림 (마이크/파일/네트워크)
│   ├── mic_client.py              # 노트북에서 실행 — 마이크 스트리밍
│   ├── start.sh                   # 파이프라인 + bore 한 번에 실행
│   └── record_samples.py          # KWS 학습 샘플 녹음 도구
│
├── Whisper/
│   ├── command_parser.py          # 명령어 파싱 / Task ID 매핑
│   ├── stt.py                     # Whisper STT
│   ├── vad.py                     # Silero VAD
│   ├── audio_loader.py            # ffmpeg 오디오 로더
│   └── config.py                  # 전체 설정값
│
├── KWS/
│   ├── kws.py                     # openWakeWord KWS 모듈
│   ├── train_oww.py               # KWS 모델 학습
│   ├── record_wakeword.py         # 웨이크워드 녹음 도구
│   ├── record_noise.py            # 환경 소음 녹음 도구
│   └── recordings/
│       ├── pos/                   # "피식아" 녹음 샘플
│       └── neg/                   # 소음 / 부정 샘플
│
├── DeepFilterNet/
│   └── denoise.py                 # DeepFilterNet3 노이즈 제거
│
├── evaluation/
│   ├── eval.py                    # STT 정확도 / 명령어 인식률 평가
│   ├── eval_kws.py                # KWS 성능 평가
│   ├── generate_tts.py            # TTS 평가 데이터 생성
│   ├── add_noise.py               # 노이즈 추가 유틸
│   └── ground_truth*.csv          # 평가 정답 데이터
│
├── pipeline/
│   └── pipeline.py                # 파일 기반 파이프라인 (테스트용)
│
├── requirements.txt               # 서버 환경 패키지
├── requirements-mic-client.txt    # 노트북 환경 패키지 (경량)
├── README.md
└── FINETUNING.md                  # KWS 재학습 가이드
```

---

## 서버 환경 설치

> Docker 환경이 이미 세팅되어 있다면 이 섹션 건너뛰기

### 1. 시스템 패키지

```bash
# Ubuntu / Debian
sudo apt install ffmpeg portaudio19-dev python3-pip
```

### 2. PyTorch 설치 (환경에 맞게 선택)

```bash
# GPU (CUDA 11.8) ← 현재 서버 환경
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# GPU (CUDA 12.1)
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.1.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

### 3. Python 패키지 설치

```bash
git clone https://github.com/ehdgur4277/Indursty_project.git
cd Indursty_project
pip install -r requirements.txt
```

### 4. KWS ONNX 모델 복사

KWS 모델 파일은 용량 문제로 git에 포함되지 않습니다. 별도로 전달받은 파일을 복사하세요.

```
KWS/oww_model/피식아.onnx
KWS/oww_model/피식아.onnx.data
```

```bash
# 예시: 다른 서버에서 복사
scp user@server:/path/to/피식아.onnx KWS/oww_model/
scp user@server:/path/to/피식아.onnx.data KWS/oww_model/
```

### 5. bore 설치 (네트워크 마이크 연결용)

```bash
cd ~
curl -L https://github.com/ekzhang/bore/releases/download/v0.5.0/bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz | tar xz
```

---

## 실행 방법

### A. 네트워크 마이크 (노트북 마이크 → 서버) ← 권장

마이크가 없는 서버(Docker 포함)에서 노트북 마이크를 TCP로 연결합니다.

#### Step 1. 서버에서 파이프라인 시작

**[연구실 컴퓨터] 터미널:**
```bash
cd /data/dhpark/Indursty_project/realtime

# 파이프라인 + bore 터널 동시 실행
./start.sh

# 노이즈 제거 없이 실행 (빠름, 권장)
./start.sh --no_denoise
```

실행하면 아래처럼 bore 포트 번호가 출력됩니다:
```
[초기화] Silero VAD 모델 로딩...
[초기화] 완료. 파이프라인 준비됨

[네트워크] TCP 대기 중 — 포트 9876
listening at bore.pub:XXXXX   ← 이 번호 메모
```

#### Step 2. 노트북에서 마이크 연결

**[노트북] 로컬 터미널 열기:**

> VSCode 사용 시: `Ctrl+Shift+P` → `Terminal: Create New Integrated Terminal (Local)` 선택

```bash
# 최초 1회 설치
pip install sounddevice numpy

# 마이크 연결 (XXXXX는 Step 1에서 나온 포트 번호)
python3 mic_client.py bore.pub --port XXXXX
```

연결 성공 시:
```
서버 bore.pub:XXXXX 연결 중...
연결됨. 마이크 스트리밍 시작 (Ctrl+C로 종료)

'피식아'라고 말해주세요.
```

---

### B. 로컬 마이크 (서버에 마이크가 직접 연결된 경우)

```bash
cd /data/dhpark/Indursty_project/realtime

python3 rt_pipeline.py
python3 rt_pipeline.py --no_denoise    # 노이즈 제거 생략 (빠름)
```

---

### C. 파일 입력 (테스트용)

```bash
cd /data/dhpark/Indursty_project/realtime

# TTS 파일로 테스트 (웨이크워드 생략)
python3 rt_pipeline.py \
    --file ../data/tts/tts_빨간색_박스_집어.wav \
    --skip_kws --no_denoise
```

---

## 노트북 연결 방법

### 사용 흐름

```
1. [연구실 컴퓨터] ./start.sh 실행
2.                  bore 포트 번호 확인 (bore.pub:XXXXX)
3. [노트북]         python3 mic_client.py bore.pub --port XXXXX
4. [노트북]         '피식아'라고 말해주세요. 뜨면 연결 완료
```

### 노트북 터미널에 표시되는 메시지

| 메시지 | 의미 |
|--------|------|
| `'피식아'라고 말해주세요.` | 웨이크워드 대기 중 |
| `[ 인식 실패 ] '피식아'라고 말해주세요.` | 다른 말로 인식됨, 재시도 |
| `[ 명령 대기 ] 명령을 말씀해주세요.` | 웨이크워드 확인, 명령어 대기 |
| `[ 인식 실패 ] 다시 말씀해주세요.` | 명령어 인식 실패, 1회 재시도 |
| `Task : TASK_PICK_RED_BOX` | 인식 성공, Task ID 출력 |

### 종료

**[연구실 컴퓨터]** 터미널에서 `Ctrl+C` → 파이프라인 + bore 동시 종료

---

## 명령어 목록

웨이크워드 **"피식아"** 이후에 아래 명령어를 말하면 됩니다.

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

> 발음이 정확하지 않아도 Levenshtein 퍼지 매칭으로 가장 가까운 명령어에 자동 매핑됩니다.

---

## 주요 설정

`Whisper/config.py` 에서 수정합니다.

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_SIZE` | `medium` | Whisper 모델 크기 (`small`로 줄이면 빠름) |
| `DEVICE` | `cuda` | 추론 장치 (`cpu` 가능) |
| `COMPUTE_TYPE` | `float16` | GPU: `float16`, CPU: `int8` |
| `SAMPLE_RATE` | `16000` | 오디오 샘플레이트 (변경 금지) |
| `VAD_THRESHOLD` | `0.5` | 발화 감지 감도 (높을수록 엄격) |
| `WAKE_WORD` | `피식아` | 웨이크워드 |
| `KWS_THRESHOLD` | `0.7` | KWS 감지 임계값 |

---

## 평가

TTS 오디오로 STT 정확도와 명령어 인식률을 평가합니다.

### 1. TTS 평가 데이터 생성

```bash
cd /data/dhpark/Indursty_project/evaluation
python3 generate_tts.py
```

`data/tts/` 에 WAV 파일 9개와 `ground_truth_tts.csv` 가 생성됩니다.

### 2. 평가 실행

```bash
cd /data/dhpark/Indursty_project/evaluation

# DFN 없이 평가 (빠름)
python3 eval.py --no_dfn

# DFN 유/무 비교 평가
python3 eval.py

# 특정 CSV로 평가
python3 eval.py --csv ground_truth_tts.csv --no_dfn
```

평가 결과 예시:
```
파일                         DFN_OFF(CER / 인식)
tts_빨간색_박스_집어.wav      CER=0.000  O(TASK_PICK_RED_BOX)
tts_파란색_박스_집어.wav      CER=0.000  O(TASK_PICK_BLUE_BOX)
...
[DFN_OFF]  평균 CER: 0.000  |  명령어 인식률: 100.0%  (9/9)
```

---

## 문제 해결

### 포트 충돌 오류 (`Address already in use`)

```bash
fuser -k 9876/tcp 2>/dev/null
fuser -k 9877/tcp 2>/dev/null
```

### 노트북 연결은 됐는데 바로 끊김

bore 터널의 초기 지연일 수 있습니다. `mic_client.py` 를 다시 실행하세요.

### "피식아" 인식이 잘 안 됨

→ [FINETUNING.md](FINETUNING.md) 의 KWS 재학습 참고

웨이크워드 패턴 추가 (`realtime/rt_pipeline.py` → `_verify_wake_word`):
```python
_WAKE_VARIANTS = [
    "피식아", "피시가", "피시카", "피싱이야", "피직아",
    "피식야", "비식아", "피식", "비싸", "pc가",
    "피시각", "이시각", "시각",
]
```

### KWS가 말 안 해도 계속 감지됨

환경 소음이 KWS 모델에 과적합된 경우입니다.  
→ [FINETUNING.md](FINETUNING.md) 의 소음 negative 샘플 추가 후 재학습 참고

### Whisper 모델 느림

`Whisper/config.py` 에서 모델 크기 변경:
```python
MODEL_SIZE = "small"     # medium → small (약 3배 빠름)
```
