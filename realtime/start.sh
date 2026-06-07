#!/bin/bash
# 파이프라인 + bore 터널 동시 실행
# Ctrl+C 하면 둘 다 같이 종료됨

cd "$(dirname "$0")"

# 혹시 남아있는 이전 프로세스 정리
fuser -k 9876/tcp 2>/dev/null

python3 rt_pipeline.py --network "$@" &
PIPELINE_PID=$!

echo "[시작] 파이프라인 PID: $PIPELINE_PID"

# Ctrl+C 시 파이프라인도 같이 종료
trap "echo '[종료] 파이프라인 중단'; kill $PIPELINE_PID 2>/dev/null; exit 0" INT TERM

~/bore local 9876 --to bore.pub

# bore가 종료되면 파이프라인도 종료
kill $PIPELINE_PID 2>/dev/null
