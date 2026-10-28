#!/usr/bin/env bash
# build-in-docker.sh — build the Sentinel OS image inside a Debian 13 container,
# so it works from any x86-64 machine with Docker (Linux, Windows/WSL2, Intel Mac).
# Takes the same options as build-iso.sh; --models/--corpus/--output paths
# anywhere on your machine are mounted into the container for you.
#
#   ./usb/build-in-docker.sh                 # desktop edition -> usb/build/out/
#   ./usb/build-in-docker.sh --lite
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v docker >/dev/null || { echo "build-in-docker: install Docker first" >&2; exit 1; }

mounts=(-v "$REPO":/work)
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --models|--corpus|--output)
            mkdir -p "$2" 2>/dev/null || true
            host="$(cd "$2" && pwd)"; inside="/mnt/${1#--}"
            mounts+=(-v "$host":"$inside"); args+=("$1" "$inside"); shift ;;
        *) args+=("$1") ;;
    esac
    shift
done

exec docker run --rm --privileged "${mounts[@]}" -w /work debian:trixie \
    bash -c 'apt-get update -qq && apt-get install -y -qq ca-certificates >/dev/null && ./usb/build-iso.sh "$@"' _ "${args[@]}"
