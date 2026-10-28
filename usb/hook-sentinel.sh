#!/bin/sh
# Runs INSIDE the image while it's being built (live-build chroot hook).
# Compiles every standard Sentinel model from the latest source, so the
# binaries always match the image's own libc. SENTINEL_TEST_ROOT lets the
# build kit test this on a normal machine.
set -e
R="${SENTINEL_TEST_ROOT:-}"
S="$R/opt/sentinel"
mkdir -p "$S/bin" "$R/usr/local/bin"

while IFS='|' read -r name bin ckpt ram cflags; do
    case "$name" in ''|'#'*) continue ;; esac
    echo "[sentinel] building $name (~${ram} MB to train): $cflags"
    # shellcheck disable=SC2086
    gcc -O2 -Wall $cflags -DCKPT_PATH="\"$ckpt\"" -o "$S/bin/$bin" "$S/src/main.c" -lm -pthread
done < "$S/models.tsv"

ln -sf /opt/sentinel/bin/sentinel "$R/usr/local/bin/sentinel"
chmod 0755 "$R/usr/local/bin/sentinel-start" "$R/usr/local/bin/sentinel-wallpaper" 2>/dev/null || true
chmod 0755 "$S/src/"*.sh 2>/dev/null || true
echo "[sentinel] built: $(ls "$S/bin" | tr '\n' ' ')"
