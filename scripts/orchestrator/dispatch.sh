#!/usr/bin/env bash
set -euo pipefail

MODE="local"
HOST=""
REPO_DIR="/home/konan147/Workspaces/SO101-Sim2Real"
PROMPT_FILE=""
MODEL="${CLAUDE_MODEL:-sonnet[1m]}"
EFFORT="${CLAUDE_EFFORT:-high}"
PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-bypassPermissions}"
TOOLS="${CLAUDE_TOOLS:-Skill, Read, Glob, Grep, Write, Edit, Bash, Agent, Monitor, TaskCreate, TaskGet, TaskList, TaskUpdate, TaskStop, WebFetch, WebSearch, Workflow, PowerShell}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/orchestrator/dispatch.sh --local --prompt-file <path>
  scripts/orchestrator/dispatch.sh --ssh <host> --repo-dir <path> --prompt-file <path>

Runs Claude Code in non-interactive JSON mode. The worker must return JSON in
the result body with keys: task_id,status,changed_files,verification,notes.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      MODE="local"
      shift
      ;;
    --ssh)
      MODE="ssh"
      HOST="${2:?missing host}"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="${2:?missing repo dir}"
      shift 2
      ;;
    --prompt-file)
      PROMPT_FILE="${2:?missing prompt file}"
      shift 2
      ;;
    --model)
      MODEL="${2:?missing model}"
      shift 2
      ;;
    --permission-mode)
      PERMISSION_MODE="${2:?missing permission mode}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PROMPT_FILE" || ! -f "$PROMPT_FILE" ]]; then
  echo "--prompt-file is required" >&2
  exit 2
fi

find_claude() {
  if command -v claude >/dev/null 2>&1; then
    command -v claude
  elif [[ -x "$HOME/.local/bin/claude.exe" ]]; then
    printf '%s\n' "$HOME/.local/bin/claude.exe"
  elif [[ -x "/c/Users/taehunkim/.local/bin/claude.exe" ]]; then
    printf '%s\n' "/c/Users/taehunkim/.local/bin/claude.exe"
  elif [[ -x "C:/Users/taehunkim/.local/bin/claude.exe" ]]; then
    printf '%s\n' "C:/Users/taehunkim/.local/bin/claude.exe"
  else
    echo "claude executable not found" >&2
    exit 127
  fi
}

if [[ "$MODE" == "local" ]]; then
  CLAUDE_BIN="$(find_claude)"
  "$CLAUDE_BIN" -p "$(cat "$PROMPT_FILE")" \
    --output-format json \
    --permission-mode "$PERMISSION_MODE" \
    --model "$MODEL" \
    --effort "$EFFORT" \
    --no-session-persistence \
    --tools "$TOOLS" \
    --allowedTools "$TOOLS"
elif [[ "$MODE" == "ssh" ]]; then
  if [[ -z "$HOST" ]]; then
    echo "--ssh requires host" >&2
    exit 2
  fi
  ssh "$HOST" "bash -lc 'cd \"$REPO_DIR\" && CLAUDE_BIN=\$(command -v claude || printf %s /home/konan147/.local/bin/claude) && \"\$CLAUDE_BIN\" -p \"\$(cat)\" --output-format json --permission-mode \"$PERMISSION_MODE\" --model \"$MODEL\" --effort \"$EFFORT\" --no-session-persistence --tools \"$TOOLS\" --allowedTools \"$TOOLS\"'" < "$PROMPT_FILE"
else
  echo "unknown mode: $MODE" >&2
  exit 2
fi
