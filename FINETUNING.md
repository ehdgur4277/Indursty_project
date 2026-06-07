# 모델 파인튜닝 가이드

KWS 웨이크워드 모델 재학습 및 Whisper STT 파인튜닝 방법

---

## 1. KWS 재학습

KWS가 오감지되거나 잘 감지되지 않을 때 재학습합니다.

### 언제 재학습해야 하나?

- 말하지 않아도 계속 "피식아" 감지됨 → negative 샘플 부족
- "피식아"라고 말해도 감지 안 됨 → positive 샘플 부족
- 환경이 바뀐 경우 (새로운 장소, 새로운 마이크)

---

### Step 1. 녹음 준비

**[연구실 컴퓨터] 터미널 1 — 녹음 서버 실행:**
```bash
fuser -k 9877/tcp 2>/dev/null
cd /data/dhpark/Indursty_project/realtime
python3 record_samples.py --type neg --count 20   # 소음 먼저
```

**[연구실 컴퓨터] 터미널 2 — bore 터널:**
```bash
~/bore local 9877 --to bore.pub
```
→ `listening at bore.pub:XXXXX` 포트 번호 확인

**[노트북]:**
```bash
python3 mic_client.py bore.pub --port XXXXX
```

---

### Step 2. 소음(negative) 녹음

```bash
# [연구실 컴퓨터] 터미널 1
python3 record_samples.py --type neg --count 20
```

- **[연구실 컴퓨터]** 에서 프롬프트 뜨면 Enter
- **[노트북]** 마이크에 **아무 말도 하지 말고** 2초 대기
- 20회 반복

저장 위치: `KWS/recordings/neg/`

---

### Step 3. 웨이크워드(positive) 녹음

```bash
# [연구실 컴퓨터] 터미널 1
python3 record_samples.py --type pos --count 30
```

- **[연구실 컴퓨터]** 에서 프롬프트 뜨면 Enter
- Enter 누르자마자 **[노트북]** 마이크에 **"피식아"** 말하기
- 30회 반복 (다양한 톤/속도로)

저장 위치: `KWS/recordings/pos/`

---

### Step 4. 재학습

```bash
cd /data/dhpark/Indursty_project/KWS
python3 train_oww.py
# 또는 epoch 수 조정
python3 train_oww.py --epochs 100
```

학습 완료 후 모델이 `KWS/oww_model/피식아.onnx`에 저장됩니다.

---

### 녹음 데이터 구조

```
KWS/recordings/
├── pos/    # "피식아" 녹음 샘플
│   ├── pos_normal_1.wav
│   ├── rec_1.wav          ← record_samples.py로 추가한 것
│   └── ...
└── neg/    # 소음 / 일반 발화
    ├── neg_silence.wav
    ├── rec_1.wav          ← record_samples.py로 추가한 것
    └── ...
```

---

### KWS 학습 파라미터 (`KWS/train_oww.py`)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `--epochs` | `60` | 학습 횟수 (높을수록 정확하지만 느림) |
| `--lr` | `1e-3` | 학습률 |
| `SNR_LIST` | `[5,10,15,20]` | 노이즈 추가 augmentation SNR |
| `SPEED_RATIOS` | `[0.85~1.15]` | 속도 augmentation 비율 |

---

## 2. Whisper STT 파인튜닝

명령어 인식률이 낮을 때 Whisper를 파인튜닝합니다.

### 데이터 준비

```bash
cd /data/dhpark/Indursty_project/Whisper/finetune

# TTS로 학습 데이터 생성
python3 prepare_dataset.py
```

`evaluation/ground_truth_tts.csv`에 정의된 명령어로 데이터셋을 생성합니다.

---

### 파인튜닝 실행

```bash
python3 train.py
```

학습 완료 후 faster-whisper 포맷으로 변환:
```bash
bash convert_to_faster_whisper.sh
```

---

### 파인튜닝 파라미터 (`Whisper/finetune/train.py`)

변환된 모델은 `Whisper/config.py`의 `MODEL_SIZE` 경로로 지정하여 사용합니다.

---

## 3. 평가

```bash
cd /data/dhpark/Indursty_project/evaluation

python3 eval.py          # STT 정확도 평가
python3 eval_kws.py      # KWS 성능 평가
python3 generate_tts.py  # TTS 평가 데이터 생성
python3 add_noise.py     # 노이즈 추가 유틸
```
