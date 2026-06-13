#!/usr/bin/env bash
#
# fetch_corpus.sh — assemble a real-world training corpus for Sentinel.
#
# Consolidates HIGH-STARRED, free, open-source security + coding repositories,
# neural-net / RNN reference implementations, and live CVE vulnerability data
# into ./corpus, then you train on the whole pile:
#
#     ./fetch_corpus.sh            # full corpus (clones several repos, ~100s of MB)
#     QUICK=1 ./fetch_corpus.sh    # small/fast demo corpus (famous CVEs + tiny repos)
#     CVE_YEARS="2023 2024 2025" ./fetch_corpus.sh
#     ./sentinel --train corpus   # <-- train the model on everything fetched
#
# 100% FREE and self-contained: no API keys, no paid services, no Claude or
# any other vendor dependency. It only uses git + curl to pull public,
# cloneable GitHub repos (works behind a github.com-only egress allowlist),
# and sentinel trains on them entirely offline afterwards. Each list below
# is meant to be edited — add your own favourite repos. .git folders are
# stripped; sentinel's consolidator skips dotfiles and caps the corpus at
# 64 MB, so over-fetching is safe.

set -uo pipefail

OUT="${1:-corpus}"
DEPTH=1
CVE_YEARS="${CVE_YEARS:-2024 2025}"     # which CVE years to sparse-checkout
# CVE_YEARS is word-split into git arguments, so allow only digits/spaces.
if ! printf '%s' "$CVE_YEARS" | grep -qE '^[0-9 ]+$'; then
  echo "error: CVE_YEARS must be space-separated years (digits only)." >&2
  exit 1
fi
mkdir -p "$OUT"

# --- curated, editable source lists -------------------------------------
SECURITY_REPOS=(
  "OWASP/CheatSheetSeries"               # defensive security cheat sheets
  "swisskyrepo/PayloadsAllTheThings"     # attack/defense technique notes
  "trimstray/the-book-of-secret-knowledge" # ops/security knowledge (markdown)
)
CODING_REPOS=(
  "TheAlgorithms/C"                      # canonical algorithms in C
  "TheAlgorithms/Python"                # canonical algorithms in Python
)
# Free, open neural-net / RNN reference implementations — so the model also
# trains on the kind of code that defines models like itself. These are how
# you'd add gated cells (LSTM/GRU), attention, Adam, etc. (no vendor lock-in).
RNN_REPOS=(
  "karpathy/micrograd"                   # tiny autograd engine
  "karpathy/char-rnn"                    # the original char-level RNN
  "karpathy/nanoGPT"                     # minimal GPT
)
# Malware-analysis knowledge (for the 'malware' retrieval topic).
MALWARE_REPOS=(
  "rshipp/awesome-malware-analysis"      # curated malware-analysis resources
  "InQuest/awesome-yara"                 # YARA detection rules/resources
)
# OSINT knowledge (for the 'osint' retrieval topic).
OSINT_REPOS=(
  "jivoi/awesome-osint"                  # the big OSINT resource list
)

# In QUICK mode swap the big lists for tiny, fast ones. CVE data already
# supplies the security content, so the quick security clone is skipped.
if [ "${QUICK:-0}" = "1" ]; then
  SECURITY_REPOS=()
  CODING_REPOS=("TheAlgorithms/C")
  RNN_REPOS=("karpathy/micrograd")
  MALWARE_REPOS=("rshipp/awesome-malware-analysis")
  OSINT_REPOS=("jivoi/awesome-osint")
fi

# --- helpers ------------------------------------------------------------
clone_repo() {
  local repo="$1" prefix="${2:-}" dest
  # validate owner/name so a crafted list entry can't traverse paths or
  # inject git arguments
  if ! printf '%s' "$repo" | grep -qE '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$'; then
    echo "  ! skip (invalid repo name)  $repo"; return
  fi
  dest="$OUT/${prefix}$(echo "$repo" | tr '/.' '__')"
  if [ -d "$dest" ]; then echo "  skip (exists)  $repo"; return; fi
  echo "  cloning  $repo"
  if timeout 240 git clone --depth "$DEPTH" --quiet "https://github.com/$repo" "$dest" 2>/dev/null; then
    rm -rf "$dest/.git"
  else
    echo "  ! failed   $repo"; rm -rf "$dest"
  fi
}

# Full CVE pull: sparse-checkout only the requested years from trickest/cve.
fetch_cves_sparse() {
  local dest="$OUT/cve_trickest"
  [ -d "$dest" ] && { echo "  skip (exists)  trickest/cve"; return; }
  echo "  sparse CVE checkout (years: $CVE_YEARS)"
  if timeout 600 git clone --depth 1 --filter=blob:none --sparse --quiet \
        https://github.com/trickest/cve "$dest" 2>/dev/null; then
    ( cd "$dest" && git sparse-checkout set $CVE_YEARS 2>/dev/null )
    rm -rf "$dest/.git"
  else
    echo "  ! CVE clone failed"; rm -rf "$dest"
  fi
}

# Quick CVE pull: just grab a handful of famous CVE writeups via raw URLs.
fetch_cves_quick() {
  local dest="$OUT/cve_famous"
  mkdir -p "$dest"
  local cves=(
    "2021/CVE-2021-44228"  "2024/CVE-2024-3094"  "2014/CVE-2014-0160"
    "2014/CVE-2014-6271"   "2017/CVE-2017-0144"  "2019/CVE-2019-0708"
    "2021/CVE-2021-34527"  "2022/CVE-2022-22965" "2023/CVE-2023-44487"
    "2020/CVE-2020-1472"
  )
  echo "  fetching ${#cves[@]} famous CVE writeups"
  for c in "${cves[@]}"; do
    local id="${c##*/}"
    curl -s --max-time 15 -o "$dest/$id.md" \
      "https://raw.githubusercontent.com/trickest/cve/main/$c.md" 2>/dev/null
  done
}

# --- run ----------------------------------------------------------------
echo "=== security repos ==="
for r in ${SECURITY_REPOS[@]+"${SECURITY_REPOS[@]}"}; do clone_repo "$r"; done
echo "=== coding repos ==="
for r in ${CODING_REPOS[@]+"${CODING_REPOS[@]}"}; do clone_repo "$r"; done
echo "=== RNN / neural-net reference repos ==="
for r in ${RNN_REPOS[@]+"${RNN_REPOS[@]}"}; do clone_repo "$r"; done
echo "=== malware-analysis knowledge ==="
for r in ${MALWARE_REPOS[@]+"${MALWARE_REPOS[@]}"}; do clone_repo "$r" "malware_"; done
echo "=== OSINT knowledge ==="
for r in ${OSINT_REPOS[@]+"${OSINT_REPOS[@]}"}; do clone_repo "$r" "osint_"; done
echo "=== CVE vulnerability data ==="
if [ "${QUICK:-0}" = "1" ]; then fetch_cves_quick; else fetch_cves_sparse; fi

echo
echo "corpus assembled in '$OUT'  (size: $(du -sh "$OUT" 2>/dev/null | cut -f1))"
echo "text/code files: $(find "$OUT" -type f \( -name '*.md' -o -name '*.c' -o -name '*.py' -o -name '*.h' -o -name '*.txt' \) 2>/dev/null | wc -l)"
echo
echo "next:  ./sentinel --train $OUT      # consolidate + train on it"
