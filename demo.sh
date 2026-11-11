#!/usr/bin/env bash
#
# demo.sh — a 30-second showcase of Sentinel's agent features.
# Builds the tiny SLM (fast), grabs a small corpus, then runs a scripted
# sequence: a CVE lookup, a malware lookup, a coding agent, and a fan-out.
#
#   ./demo.sh
#
set -uo pipefail
cd "$(dirname "$0")" || exit 1

B='\033[36m'; G='\033[32m'; D='\033[90m'; R='\033[0m'
say() { printf "${B}── %s${R}\n" "$*"; }

command -v gcc >/dev/null || { echo "need gcc"; exit 1; }

say "building the tiny SLM (~1.2M params, builds in a second)"
# LayerNorm is the default build (as in the Makefile). A checkpoint trained by
# an older, plain-GRU build ("SLM3" tag) keeps its matching binary instead.
ln_flag() {
  if [ -f "$1" ] && [ "$(od -An -tu4 -N4 "$1" 2>/dev/null | tr -d ' ')" = 1397509427 ]; then echo ""
  else echo "-DUSE_LN"; fi
}
# shellcheck disable=SC2046
[ -x ./sentinel-slm ] || gcc -O2 -Wall $(ln_flag sentinel-slm.bin) -DHIDDEN=256 -DEMBED=64 -DNUM_LAYERS=2 \
    -DCKPT_PATH='"sentinel-slm.bin"' -o sentinel-slm main.c -lm -pthread

if [ ! -d corpus ]; then
  say "fetching a small demo corpus (famous CVEs + malware/OSINT notes)"
  QUICK=1 ./fetch_corpus.sh corpus >/dev/null 2>&1 || echo "  (offline — retrieval will be limited)"
fi

say "driving Sentinel through a few tasks (watch the agents spawn)"
printf '%s\n' \
  'show me a cve about http' \
  'TASK: list files in this directory' \
  'TASK: write a C function to reverse a string' \
  'FANOUT: 12 report current date' \
  | SLM_EPOCHS=20 ./sentinel-slm 2>&1 \
  | sed -n '/=== read loop ===/,$p'

printf "\n${G}That's Sentinel — a local AI security agent, no cloud, no API keys.${R}\n"
printf "${D}Try the interfaces:  ./sentinel --tui   ·   ./sentinel --serve 8080${R}\n"
