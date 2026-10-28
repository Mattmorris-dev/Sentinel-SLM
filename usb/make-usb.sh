#!/usr/bin/env bash
# make-usb.sh — write a Sentinel OS image to a USB stick, add a persistence
# partition (so it keeps learning across reboots), and optionally load YOUR
# trained models + corpus onto that stick (this is what makes it the paid edition).
#
#   sudo ./usb/make-usb.sh <image.iso> /dev/sdX [--models DIR] [--corpus DIR]
#                          [--no-persistence] [--dry-run] [--yes] [--force]
#
# !! ERASES THE ENTIRE TARGET DEVICE. It refuses any disk your system is running
#    from, refuses non-USB disks unless --force, and asks you to type the device
#    name to confirm (unless --yes). --dry-run prints what it would do.
set -euo pipefail

ISO=""; DEV=""; MODELS=""; CORPUS=""; PERSIST=1; DRY=0; YES=0; FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --models) MODELS="$2"; shift ;;
        --corpus) CORPUS="$2"; shift ;;
        --no-persistence) PERSIST=0 ;;
        --dry-run) DRY=1 ;;
        --yes) YES=1 ;;
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        -*) echo "make-usb: unknown option '$1'" >&2; exit 1 ;;
        *) if [ -z "$ISO" ]; then ISO="$1"; elif [ -z "$DEV" ]; then DEV="$1"; else echo "make-usb: extra argument '$1'" >&2; exit 1; fi ;;
    esac
    shift
done

die() { echo "make-usb: $*" >&2; exit 1; }
run() { if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; else "$@"; fi; }

[ -n "$ISO" ] && [ -n "$DEV" ] || die "usage: make-usb.sh <image.iso> /dev/sdX  (see --help)"
[ -f "$ISO" ] || die "no such image: $ISO"
[ "$DRY" = 1 ] || [ "$(id -u)" = 0 ] || die "run as root (sudo)"
[ -b "$DEV" ] || die "$DEV is not a block device"
t="$(lsblk -dno TYPE "$DEV")"
{ [ "$t" = disk ] || { [ "$t" = loop ] && [ "$FORCE" = 1 ]; }; } || die "$DEV is a $t, not a whole disk — give the whole stick (e.g. /dev/sdb, not /dev/sdb1)"
[ -z "$MODELS" ] || [ -d "$MODELS" ] || die "--models: no such directory '$MODELS'"
[ -z "$CORPUS" ] || [ -d "$CORPUS" ] || die "--corpus: no such directory '$CORPUS'"

# --- never touch a disk the running system uses
while read -r name mnts; do
    for m in $mnts; do
        case "$m" in
            /|/boot|/boot/*|/home|/usr|/var|"[SWAP]")
                die "$DEV holds $m of THIS system ($name) — refusing" ;;
        esac
    done
done < <(lsblk -lnpo NAME,MOUNTPOINTS "$DEV" 2>/dev/null || lsblk -lnpo NAME,MOUNTPOINT "$DEV")
rootsrc="$(findmnt -no SOURCE / 2>/dev/null || true)"
if [ -b "$rootsrc" ]; then
    rootdisk="$(lsblk -no PKNAME "$rootsrc" 2>/dev/null | head -1)"
    rootdisk="${rootdisk:+/dev/$rootdisk}"; rootdisk="${rootdisk:-$rootsrc}"
    [ "$(readlink -f "$DEV")" != "$(readlink -f "$rootdisk")" ] || die "$DEV is the disk this system boots from — refusing"
fi

# --- should look like a USB stick
rm_flag="$(lsblk -dno RM "$DEV" | tr -d ' ')"; tran="$(lsblk -dno TRAN "$DEV" | tr -d ' ')"
if [ "$rm_flag" != 1 ] && [ "$tran" != usb ] && [ "$FORCE" = 0 ]; then
    die "$DEV doesn't look like a USB stick (removable=$rm_flag, transport=${tran:-?}) — use --force if you're sure"
fi

iso_bytes="$(stat -c %s "$ISO")"; dev_bytes="$(lsblk -bdno SIZE "$DEV")"
need=$(( iso_bytes + (PERSIST == 1 ? 1024*1024*1024 : 0) ))
[ "$dev_bytes" -ge "$need" ] || die "$DEV is too small ($((dev_bytes/1000000)) MB; need $((need/1000000)) MB)"

echo "About to ERASE this device and write $(basename "$ISO"):"
lsblk -o NAME,SIZE,MODEL,TRAN,RM "$DEV"
if [ "$YES" = 0 ] && [ "$DRY" = 0 ]; then
    read -r -p "Type the device name ($DEV) to continue: " answer
    [ "$answer" = "$DEV" ] || die "not confirmed — nothing written"
fi

# --- write
for p in $(lsblk -lnpo NAME "$DEV" | tail -n +2); do run umount "$p" 2>/dev/null || true; done
echo "writing image (this takes a few minutes)..."
run dd if="$ISO" of="$DEV" bs=4M status=progress conv=fsync

if [ "$PERSIST" = 1 ]; then
    echo "adding the persistence partition in the free space..."
    run partprobe "$DEV" || true; [ "$DRY" = 1 ] || sleep 2
    if [ "$DRY" = 1 ]; then
        echo "  [dry-run] echo ',,L' | sfdisk --append --no-reread $DEV"
        part="${DEV}3"
    else
        echo ',,L' | sfdisk --append --no-reread "$DEV" >/dev/null \
            || die "couldn't add the partition automatically — the stick still boots (without persistence)"
        partprobe "$DEV" || true; sleep 2
        part="$(lsblk -lnpo NAME,TYPE "$DEV" | awk '$2=="part"{p=$1} END{print p}')"
    fi
    run mkfs.ext4 -F -q -L persistence "$part"
    mnt="$(mktemp -d)"
    run mount "$part" "$mnt"
    if [ "$DRY" = 1 ]; then echo "  [dry-run] write '/home union' to $mnt/persistence.conf"
    else echo "/home union" > "$mnt/persistence.conf"; fi
    if [ -n "$MODELS" ]; then
        run mkdir -p "$mnt/sentinel-models"
        for f in "$MODELS"/*.bin; do [ -f "$f" ] && run cp "$f" "$mnt/sentinel-models/"; done
        echo "loaded trained models: $(ls "$MODELS"/*.bin 2>/dev/null | wc -l) file(s)"
    fi
    if [ -n "$CORPUS" ]; then
        run cp -r "$CORPUS" "$mnt/sentinel-corpus"
        echo "loaded corpus: $(du -sh "$CORPUS" | cut -f1)"
    fi
    run chmod -R a+rX "$mnt"
    run umount "$mnt"; rmdir "$mnt" 2>/dev/null || true
elif [ -n "$MODELS$CORPUS" ]; then
    echo "note: --models/--corpus need the persistence partition; skipped (--no-persistence)"
fi
run sync
echo "done — boot a PC from $DEV (pick it in the boot menu, often F12/F11/Esc)."
