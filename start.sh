#!/usr/bin/env bash
#
# start.sh — one-command launcher for Sentinel.
#
# Does the whole pipeline, skipping any step that's already done:
#   1. build ./sentinel (if missing or main.c changed)
#   2. fetch a training corpus (if ./corpus is missing)
#   3. train on the corpus and save a checkpoint (if no checkpoint yet)
#   4. launch the interactive agent loop
#
# Usage:
#   ./start.sh                 # build, quick-fetch, train once, run
#   FULL=1 ./start.sh          # fetch the FULL corpus (big: many repos + CVE years)
#   RETRAIN=1 ./start.sh       # force a fresh training pass before running
#   SLM_EPOCHS=20000 ./start.sh# train harder
#   ./start.sh --no-run        # set everything up but don't enter the loop

set -uo pipefail
cd "$(dirname "$0")"

log() { printf '\033[36m[start]\033[0m %s\n' "$*"; }

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<'EOF'
start.sh — one-command launcher for Sentinel (build + fetch + train + run).

  ./start.sh                 build, quick-fetch a corpus, train once, then run
  FULL=1 ./start.sh          fetch the FULL corpus (many repos + CVE years)
  RETRAIN=1 ./start.sh       force a fresh training pass before running
  SLM_EPOCHS=20000 ./start.sh  train harder
  ./start.sh --no-run        set everything up but don't enter the loop
  ./start.sh --help          show this help

Once running, type 'TASK: ...' to spawn an agent, ask 'show me a cve about X',
or just type text to teach it. Run './sentinel --help' for the full AI guide.
EOF
  exit 0
fi

# --- 1. build ----------------------------------------------------------
if ! command -v gcc >/dev/null 2>&1; then
  echo "error: gcc not found. Install it (e.g. 'sudo apt install build-essential')." >&2
  exit 1
fi
if [ ! -x ./sentinel ] || [ main.c -nt ./sentinel ]; then
  log "building ./sentinel ..."
  gcc -O2 -Wall -o sentinel main.c -lm || { echo "build failed" >&2; exit 1; }
else
  log "binary up to date — skipping build."
fi

# --- 2. fetch corpus ---------------------------------------------------
if [ ! -d corpus ]; then
  if [ "${FULL:-0}" = "1" ]; then
    log "fetching FULL corpus (this can take a while)..."
    ./fetch_corpus.sh corpus || log "fetch reported issues — continuing anyway."
  else
    log "fetching QUICK corpus (famous CVEs + a couple repos)..."
    QUICK=1 ./fetch_corpus.sh corpus || log "fetch reported issues — continuing anyway."
  fi
else
  log "corpus/ already present — skipping fetch (delete it to refetch)."
fi

# --- 3. train ----------------------------------------------------------
if [ ! -f sentinel.bin ] || [ "${RETRAIN:-0}" = "1" ]; then
  if [ -d corpus ]; then
    log "training on corpus/ ..."
    ./sentinel --train corpus || log "training reported issues — continuing."
  else
    log "no corpus to train on — the model will use its built-in bootstrap text."
  fi
else
  log "checkpoint sentinel.bin exists — skipping training (RETRAIN=1 to redo)."
fi

# --- 4. run ------------------------------------------------------------
if [ "${1:-}" = "--no-run" ]; then
  log "setup complete (--no-run). Launch later with: ./sentinel"
  exit 0
fi

log "launching Sentinel. Type text to teach it, 'TASK: ...' to spawn an agent,"
log "or ask a CVE question (e.g. 'show me a cve about android'). Ctrl-D to quit."
# Keep the interactive warm-up short since we already trained above.
export SLM_EPOCHS="${SLM_EPOCHS:-150}"
exec ./sentinel
