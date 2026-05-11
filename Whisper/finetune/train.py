"""
Whisper 파인튜닝 스크립트
openai/whisper-medium → 한국어 로봇 명령어 특화

설치:
    pip install transformers datasets accelerate evaluate jiwer soundfile librosa

사용법:
    python train.py
    python train.py --model_size small   # GPU 메모리 적을 때
    python train.py --epochs 10

학습 완료 후 faster-whisper 변환:
    ct2-transformers-converter --model ./output/checkpoint-best --output_dir ./output/ct2 --quantization float16
"""

import argparse
import csv
import numpy as np
import torch
import soundfile as sf
import librosa
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset, Audio
import evaluate

# =============================================================================
# 설정
# =============================================================================

DATASET_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR  = Path(__file__).parent / "output"
SAMPLE_RATE = 16000

# =============================================================================
# CSV → HuggingFace Dataset
# =============================================================================

def load_csv_dataset(csv_path: Path) -> Dataset:
    records = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({"audio_path": row["audio"], "text": row["text"]})
    return Dataset.from_list(records)


def preprocess(batch, processor):
    audio, sr = sf.read(batch["audio_path"])
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = audio.astype(np.float32)

    inputs = processor.feature_extractor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
    )
    batch["input_features"] = inputs.input_features[0]

    with processor.as_target_processor():
        labels = processor.tokenizer(batch["text"]).input_ids
    batch["labels"] = labels
    return batch

# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # BOS 토큰 제거 (Whisper는 decoder_start_token_id로 시작)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# =============================================================================
# 메인
# =============================================================================

def main(args):
    model_name = f"openai/whisper-{args.model_size}"
    print(f"\n[모델] {model_name}")

    processor = WhisperProcessor.from_pretrained(model_name, language="Korean", task="transcribe")
    model     = WhisperForConditionalGeneration.from_pretrained(model_name)

    # 한국어 강제 설정
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    model.config.suppress_tokens = []

    # 데이터셋 로드
    train_csv = DATASET_DIR / "train.csv"
    test_csv  = DATASET_DIR / "test.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv 없음. 먼저 prepare_dataset.py 실행하세요.")

    print("[데이터] 로드 중...")
    train_ds = load_csv_dataset(train_csv)
    test_ds  = load_csv_dataset(test_csv)

    preprocess_fn = lambda b: preprocess(b, processor)
    train_ds = train_ds.map(preprocess_fn, remove_columns=["audio_path", "text"])
    test_ds  = test_ds.map(preprocess_fn,  remove_columns=["audio_path", "text"])
    print(f"  train: {len(train_ds)}개 / test: {len(test_ds)}개")

    # 평가 지표 (CER - 한국어는 CER이 더 적합)
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    # 학습 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=128,
        logging_steps=10,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=processor.feature_extractor,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
    )

    print("\n[학습] 시작...")
    trainer.train()

    best_dir = OUTPUT_DIR / "checkpoint-best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    print(f"\n[완료] 모델 저장: {best_dir}")
    print("\n[다음 단계] faster-whisper 변환:")
    print(f"  ct2-transformers-converter --model {best_dir} --output_dir {OUTPUT_DIR}/ct2 --quantization float16")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="medium", choices=["tiny", "base", "small", "medium", "large-v2"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)
