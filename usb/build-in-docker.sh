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
have_work=0
while [ $# -gt 0 ]; do
    case "$1" in
        --models|--corpus|--output)
            [ $# -ge 2 ] || { echo "build-in-docker: $1 needs a directory" >&2; exit 1; }
            if [ "$1" = --output ]; then mkdir -p "$2"
            elif [ ! -d "$2" ]; then echo "build-in-docker: $1: no such directory '$2'" >&2; exit 1; fi
            host="$(cd "$2" && pwd)"; inside="/mnt/${1#--}"
            mounts+=(-v "$host":"$inside"); args+=("$1" "$inside"); shift ;;
        --work) have_work=1; args+=("$1") ;;
        *) args+=("$1") ;;
    esac
    shift
done
# Build the Debian tree on the container's own disk, not on the bind-mounted
# repo: a chroot needs device nodes and root ownership, which Docker Desktop
# mounts (and nodev home dirs) don't allow. Only the finished image comes back,
# owned by whoever ran this script.
[ "$have_work" = 1 ] || args+=(--work /var/tmp/sentinel-lb)

exec docker run --rm --privileged "${mounts[@]}" -w /work \
    -e HOST_OWNER="$(id -u):$(id -g)" debian:trixie \
    bash -c 'apt-get update -qq && apt-get install -y -qq ca-certificates >/dev/null || exit 1
             ./usb/build-iso.sh "$@"; rc=$?
             for d in /work/usb/build /mnt/output; do
                 if [ -d "$d" ]; then chown -R "$HOST_OWNER" "$d" || true; fi
             done
             exit $rc' _ "${args[@]}"
