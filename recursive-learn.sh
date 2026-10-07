#!/usr/bin/env bash
# Recursive self-learning loop for Sentinel.
#
# Each round the model (1) GENERATES text from its current checkpoint, then
# (2) RETRAINS on that text MIXED WITH the real corpus. The real corpus is the
# anchor: training purely on a model's own output causes "model collapse" (it
# drifts into gibberish), so the corpus must dominate every round. This is a
# reinforcement / practice loop — it sharpens what the model already knows, it
# does NOT invent knowledge the corpus doesn't contain.
#
# Usage:
#   ./recursive-learn.sh                 # 5 rounds on ./corpus (or built-in)
#   ROUNDS=10 SLM_EPOCHS=4000 ./recursive-learn.sh
#   CORPUS=./my_data BIN=./sentinel-ln ./recursive-learn.sh
#
# Runs on any node (scp it to a Pi / mini-PC / laptop and run it there).
set -euo pipefail

BIN="${BIN:-./sentinel}"
CORPUS="${CORPUS:-corpus}"          # real-data anchor (file or dir)
ROUNDS="${ROUNDS:-5}"
GEN_CHARS="${GEN_CHARS:-2000}"      # how much the model generates per round
EPOCHS="${SLM_EPOCHS:-2000}"        # training iters per round
SELF="${SELF:-self-corpus.txt}"     # accumulated self-generated text
MAXSELF="${MAXSELF:-200000}"        # cap self-corpus size (bytes)

[ -x "$BIN" ] || { echo "recursive-learn: no executable '$BIN' (run make first)" >&2; exit 1; }
if [ ! -e "$CORPUS" ]; then
    echo "recursive-learn: anchor corpus '$CORPUS' not found — using the built-in corpus as the anchor." >&2
    CORPUS=""
fi

for r in $(seq 1 "$ROUNDS"); do
    echo "== recursive round $r/$ROUNDS =="
    # 1. generate from the current model (pure inference, no training)
    "$BIN" --generate "$GEN_CHARS" >> "$SELF" 2>/dev/null || true
    # keep the self-corpus bounded (most recent MAXSELF bytes)
    if [ -f "$SELF" ]; then tail -c "$MAXSELF" "$SELF" > "$SELF.tmp" && mv "$SELF.tmp" "$SELF"; fi
    echo "   generated $GEN_CHARS chars (self-corpus now $(wc -c < "$SELF") bytes)"
    # 2. retrain, anchored on the real corpus + the self-samples, then checkpoint
    if [ -n "$CORPUS" ]; then
        SLM_EPOCHS="$EPOCHS" "$BIN" --train "$CORPUS" "$SELF" | tail -2
    else
        SLM_EPOCHS="$EPOCHS" "$BIN" --train "$SELF" | tail -2
    fi
done

echo "recursive-learn: done — $ROUNDS rounds, model checkpointed. self-corpus: $SELF"
