#!/bin/bash
# 파인튜닝된 HuggingFace Whisper → faster-whisper(CTranslate2) 변환
#
# 설치: pip install ctranslate2
# 사용: bash convert_to_faster_whisper.sh

INPUT_DIR="$(dirname "$0")/output/checkpoint-best"
OUTPUT_DIR="$(dirname "$0")/output/ct2"

echo "[변환] $INPUT_DIR → $OUTPUT_DIR"

ct2-transformers-converter \
    --model "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --quantization float16 \
    --force

echo ""
echo "[완료] 변환된 모델: $OUTPUT_DIR"
echo ""
echo "realtime_pipeline.py에서 모델 경로 변경:"
echo "  MODEL_SIZE = \"$OUTPUT_DIR\""
