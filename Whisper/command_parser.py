"""
STT 결과 → 가장 가까운 유효 명령어 매핑 → Task ID

흐름:
  raw STT text
    → normalize_korean_command()  (공백 정리 등 경량 전처리)
    → fuzzy_match()               (Levenshtein으로 정답 명령어 중 최근접)
    → parse_command()             (매핑된 정답 문자열에서 Task ID 추출)
"""

# =============================================================================
# 정답 명령어 목록 (이것만 관리하면 됨)
# =============================================================================

VALID_COMMANDS = [
    "빨간 통 집어",
    "노란 통 집어",
    "박스 집어",
    "왼쪽 빨간 통 집어",
    "오른쪽 빨간 통 집어",
    "왼쪽 노란 통 집어",
    "오른쪽 노란 통 집어",
    "빨간 통 왼쪽으로 집어",
    "빨간 통 오른쪽으로 집어",
    "노란 통 왼쪽으로 집어",
    "노란 통 오른쪽으로 집어",
    "빨간 통 옮겨",
    "노란 통 옮겨",
    "박스 옮겨",
]

# 명령어 → Task ID 직접 매핑
COMMAND_TO_TASK = {
    "빨간 통 집어":           "TASK_PICK_RED",
    "노란 통 집어":           "TASK_PICK_YELLOW",
    "박스 집어":              "TASK_PICK_BOX",
    "왼쪽 빨간 통 집어":      "TASK_PICK_RED_LEFT",
    "오른쪽 빨간 통 집어":    "TASK_PICK_RED_RIGHT",
    "왼쪽 노란 통 집어":      "TASK_PICK_YELLOW_LEFT",
    "오른쪽 노란 통 집어":    "TASK_PICK_YELLOW_RIGHT",
    "빨간 통 왼쪽으로 집어":  "TASK_PICK_RED_LEFT",
    "빨간 통 오른쪽으로 집어":"TASK_PICK_RED_RIGHT",
    "노란 통 왼쪽으로 집어":  "TASK_PICK_YELLOW_LEFT",
    "노란 통 오른쪽으로 집어":"TASK_PICK_YELLOW_RIGHT",
    "빨간 통 옮겨":           "TASK_MOVE_RED",
    "노란 통 옮겨":           "TASK_MOVE_YELLOW",
    "박스 옮겨":              "TASK_MOVE_BOX",
}

FUZZY_THRESHOLD = 0.5  # 유사도 이 미만이면 매핑 실패로 처리

# =============================================================================
# 경량 전처리 (공백 정리만 — 오인식 패턴 등록 불필요)
# =============================================================================

def normalize_korean_command(text: str) -> str:
    return " ".join(text.strip().split())

# =============================================================================
# Levenshtein 편집거리
# =============================================================================

def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]

# =============================================================================
# 퍼지 매칭: STT 결과 → 가장 가까운 정답 명령어
# =============================================================================

def fuzzy_match(text: str) -> tuple[str | None, float]:
    """
    VALID_COMMANDS 중 편집거리가 가장 짧은 명령어를 반환.
    공백 제거 후 비교하므로 띄어쓰기 차이에 강건함.
    반환: (matched_command or None, similarity_score)
    """
    text_c = text.replace(" ", "")
    best_cmd, best_score = None, -1.0

    for cmd in VALID_COMMANDS:
        cmd_c = cmd.replace(" ", "")
        dist  = _levenshtein(text_c, cmd_c)
        score = 1.0 - dist / max(len(text_c), len(cmd_c), 1)
        if score > best_score:
            best_score = score
            best_cmd   = cmd

    if best_score >= FUZZY_THRESHOLD:
        return best_cmd, best_score
    return None, best_score

# =============================================================================
# 파싱 (fuzzy match → Task ID)
# =============================================================================

def parse_command(text: str) -> dict:
    matched_cmd, score = fuzzy_match(text)

    if matched_cmd is None:
        return {
            "raw": text,
            "matched": None,
            "similarity": round(score, 2),
            "task_id": "TASK_UNKNOWN",
            "status": "FAIL",
            "reason": f"유사 명령어 없음 (최고 유사도 {score:.2f})",
        }

    task_id = COMMAND_TO_TASK[matched_cmd]
    return {
        "raw": text,
        "matched": matched_cmd,
        "similarity": round(score, 2),
        "task_id": task_id,
        "status": "SUCCESS",
        "reason": "success",
    }
