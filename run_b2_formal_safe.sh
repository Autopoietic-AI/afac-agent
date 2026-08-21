#!/usr/bin/env bash

set -uo pipefail

REPAIR_COMMIT="ca2a803533bb120eb3a88f2404fbea99a5654147"
B2_ROOT="C:/Users/李天皓/agent比赛/B推荐/B推荐"
OUT_ROOT="artifacts/v2_formal_runs/b2"
LOG_FILE="${OUT_ROOT}/b2_v21_formal_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "AFAC B2 FORMAL PREFLIGHT"
echo "Time: $(date)"
echo "============================================================"

cd "/c/Users/李天皓/agent比赛/agent/AFAC_AGENT_V1_1_CODEX_READY/afac_agent_v1" || {
  echo "ERROR: 无法进入仓库"
  return 1 2>/dev/null || true
}

# 加载AFAC专用配置
if [ ! -f "$HOME/.afac/llm.env" ]; then
  echo "ERROR: 缺少 $HOME/.afac/llm.env"
  return 1 2>/dev/null || true
fi

source "$HOME/.afac/llm.env"

# 不使用通用OpenAI变量
unset OPENAI_API_KEY
unset OPENAI_BASE_URL

echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"

# 检查修复提交
if ! git merge-base --is-ancestor "$REPAIR_COMMIT" HEAD; then
  echo "ERROR: 当前HEAD不包含科学执行修复：$REPAIR_COMMIT"
  return 1 2>/dev/null || true
fi

echo "Science repair: PRESENT"

# 只检查已跟踪文件修改；允许未跟踪的artifacts日志存在
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: 检测到已跟踪代码存在未提交修改"
  git status --short
  return 1 2>/dev/null || true
fi

echo "Tracked source tree: CLEAN"

# 检查数据目录
if [ ! -d "$B2_ROOT" ]; then
  echo "ERROR: B2数据目录不存在：$B2_ROOT"
  return 1 2>/dev/null || true
fi

# 检查LLM配置
python - <<'PY'
import os

key = os.getenv("DASHSCOPE_API_KEY", "")
base = os.getenv("AFAC_LLM_BASE_URL", "")
model = os.getenv("AFAC_LLM_MODEL", "")

print({
    "api_key_ready": bool(key),
    "base_url": base,
    "configured_model": model or "<project default>",
})

if not key:
    raise SystemExit("ERROR: DASHSCOPE_API_KEY 缺失")

if not base:
    raise SystemExit("ERROR: AFAC_LLM_BASE_URL 缺失")

if any(c in key for c in ("'", '"')):
    raise SystemExit("ERROR: API Key包含引号")

print("LLM CONFIG: PASS")
PY

if [ $? -ne 0 ]; then
  echo "ERROR: LLM配置预检失败"
  return 1 2>/dev/null || true
fi

mkdir -p "$OUT_ROOT"

echo
echo "============================================================"
echo "STARTING AFAC B2 FORMAL RUN"
echo "Data:    $B2_ROOT"
echo "Output:  $OUT_ROOT"
echo "Budget:  7200 seconds"
echo "Reserve: 600 seconds"
echo "Log:     $LOG_FILE"
echo "============================================================"
echo

python -u -m afac_agent.main v2-run \
  --task B2 \
  --data-root "$B2_ROOT" \
  --out-root "$OUT_ROOT" \
  --max-wall-clock-seconds 7200 \
  --deployment-reserve-seconds 600 \
  --require-llm \
  --force-new-execution \
  2>&1 | tee "$LOG_FILE"

PYTHON_EXIT=${PIPESTATUS[0]}

echo
echo "============================================================"
echo "AFAC B2 FORMAL RUN ENDED"
echo "Time: $(date)"
echo "Python exit code: $PYTHON_EXIT"
echo "Log: $LOG_FILE"
echo "============================================================"

exit "$PYTHON_EXIT"
