# Industry Voice Command Pipeline

음성 명령으로 로봇 Task ID를 전달하는 실시간 파이프라인

---

## 동작 흐름

```
마이크 입력
    │
    ▼
  VAD (묵음 필터)
    │ 발화 감지 시
    ▼
  KWS ("피식아" 감지)
    │ 웨이크워드 확인 시
    ▼
  VAD (발화 종료 감지)
    │
    ▼
  DeepFilterNet (노이즈 제거, 선택)
    │
    ▼
  Whisper STT (한국어 인식)
    │
    ▼
  퍼지 매칭 → Task ID → 로봇 전달
```

---

## 설치

### 시스템 패키지
```bash
# Ubuntu / Debian
sudo apt install ffmpeg portaudio19-dev python3-pip
```

### PyTorch
```bash
# GPU (CUDA 11.8)
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch==2.1.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

### 패키지
```bash
pip install -r requirements.txt
```

---

## 실행 방법

### A. 로컬 마이크 (서버에 마이크 직접 연결)

```bash
cd realtime
python3 rt_pipeline.py
python3 rt_pipeline.py --no_denoise    # 노이즈 제거 생략 (빠름)
```

---

### B. 네트워크 마이크 — 노트북 마이크를 서버로 연결 ★

**[연구실 컴퓨터] 터미널 1 — 파이프라인 + bore 실행:**
```bash
cd /data/dhpark/Indursty_project/realtime
./start.sh
# 또는
./start.sh --no_denoise
```
실행하면 아래처럼 bore 포트 번호가 출력됩니다:
```
listening at bore.pub:XXXXX  ← 이 번호 메모
```

**[노트북] 로컬 터미널 열기:**

> VSCode 사용 시: `Ctrl+Shift+P` → `Terminal: Create New Integrated Terminal (Local)`

```bash
# 최초 1회 설치
pip install sounddevice numpy

# 마이크 스트리밍 시작
python3 mic_client.py bore.pub --port XXXXX
```

---

### C. 파일 입력 (테스트용)

```bash
cd realtime
python3 rt_pipeline.py --file ../data/Red_noisy.m4a --skip_kws --no_denoise
```

---

## 사용 방법

1. `./start.sh` 실행 → bore 포트 번호 확인
2. **[노트북]** `python3 mic_client.py bore.pub --port XXXXX` 실행
3. **"피식아"** 라고 말하기
4. 바로 이어서 명령어 말하기
5. 결과가 **[노트북] 터미널에도 출력**됨
6. 다시 대기 상태 → 3번부터 반복

**종료:** `Ctrl+C` (파이프라인 + bore 동시 종료)

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
| `DEVICE` | `cuda` | 추론 장치 |
| `VAD_THRESHOLD` | `0.5` | 발화 감지 감도 |
| `WAKE_WORD` | `피식아` | 웨이크워드 |
| `KWS_THRESHOLD` | `0.7` | 웨이크워드 감지 임계값 |

---

> KWS 재학습, Whisper 파인튜닝은 [FINETUNING.md](FINETUNING.md) 참고
