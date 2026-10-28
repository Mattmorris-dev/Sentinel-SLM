#!/usr/bin/env bash
# build-iso.sh — build the bootable Sentinel OS image: Debian + every Sentinel
# model + the agents, as one hybrid ISO you can write to any USB stick (boots
# BIOS and UEFI x86-64 PCs).
#
# Run as root on an x86-64 Debian 13 machine, or anywhere with Docker via
#   ./usb/build-in-docker.sh [options]
#
#   --lite          no desktop: boots straight into the Sentinel TUI (~0.8 GB)
#                   (default: XFCE desktop + browser GUI, ~2 GB)
#   --with-source   also produce the Debian source tarball (GPL compliance when selling)
#   --models DIR    bake trained checkpoints into the image (big! usually better to
#                   add them per-stick with make-usb.sh --models)
#   --corpus DIR    bake a corpus into the image
#   --output DIR    where the .iso goes (default usb/build/out)
#   --stage-only    prepare the build tree but don't run live-build (testing)
#
# Edit usb/brand.conf to rebrand.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# shellcheck source=usb/brand.conf
. "$HERE/brand.conf"

LITE=0; WITH_SOURCE=0; STAGE_ONLY=0
OUT="$HERE/build/out"; WORK="$HERE/build/lb"; MODELS=""; CORPUS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --lite)        LITE=1 ;;
        --with-source) WITH_SOURCE=1 ;;
        --stage-only)  STAGE_ONLY=1 ;;
        --models)      MODELS="$2"; shift ;;
        --corpus)      CORPUS="$2"; shift ;;
        --output)      OUT="$2"; shift ;;
        --work)        WORK="$2"; shift ;;
        -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "build-iso: unknown option '$1' (see --help)" >&2; exit 1 ;;
    esac
    shift
done

die() { echo "build-iso: $*" >&2; exit 1; }
say() { echo "==> $*"; }

VERSION="$(grep -m1 '#define VERSION' "$REPO/main.c" | grep -o '"[^"]*"' | tr -d '"')"
[ -n "$VERSION" ] || die "couldn't read VERSION from main.c"
EDITION="$([ "$LITE" = 1 ] && echo lite || echo desktop)"
IMAGE_NAME="${OS_ID}-os-${VERSION}-${EDITION}"
PRO=0; { [ -n "$MODELS" ] || [ -n "$CORPUS" ]; } && PRO=1 && IMAGE_NAME="${IMAGE_NAME}-pro"

# ---------------------------------------------------------------- preflight
if [ "$STAGE_ONLY" = 0 ]; then
    [ "$(id -u)" = 0 ] || die "run as root (sudo), or use ./usb/build-in-docker.sh"
    arch="$(dpkg --print-architecture 2>/dev/null || echo unknown)"
    [ "$arch" = amd64 ] || die "build on an x86-64 machine (this is '$arch'); an image built on a Pi only boots ARM boards"
    avail_kb="$(df -Pk "$HERE" | awk 'NR==2{print $4}')"
    [ "$avail_kb" -gt 12000000 ] || die "need ~12 GB free disk for the build"
    say "installing build tools"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq live-build debootstrap xorriso squashfs-tools mtools dosfstools \
        grub-efi-amd64-bin grub-pc-bin syslinux syslinux-common isolinux librsvg2-bin \
        fonts-dejavu-core >/dev/null
fi
command -v rsvg-convert >/dev/null || die "need rsvg-convert (apt-get install librsvg2-bin)"
[ -f "$REPO/$MARK" ] || die "brand mark not found: $MARK (set MARK in usb/brand.conf)"
[ -z "$MODELS" ] || [ -d "$MODELS" ] || die "--models: no such directory '$MODELS'"
[ -z "$CORPUS" ] || [ -d "$CORPUS" ] || die "--corpus: no such directory '$CORPUS'"

say "building $OS_NAME $VERSION ($EDITION$([ "$PRO" = 1 ] && echo ', PRO'))"
[ "$PRO" = 1 ] && echo "    !! this image contains your trained models/corpus — do NOT publish it"

# ---------------------------------------------------------------- live-build config
mkdir -p "$WORK" "$OUT"
cd "$WORK"
if [ "$STAGE_ONLY" = 0 ]; then
    lb clean --purge >/dev/null 2>&1 || true
    src_opts=()
    [ "$WITH_SOURCE" = 1 ] && src_opts=(--source true --source-images tar)
    lb config \
        --mode debian \
        --distribution "$DEBIAN_RELEASE" \
        --archive-areas "main contrib non-free non-free-firmware" \
        --apt-recommends false \
        --firmware-binary false --firmware-chroot false \
        --memtest none \
        --image-name "$IMAGE_NAME" \
        --iso-application "$OS_NAME" \
        --iso-publisher "$PUBLISHER" \
        --iso-volume "$ISO_VOLUME" \
        --bootappend-live "boot=live components persistence username=$OS_ID hostname=$OS_ID quiet splash" \
        "${src_opts[@]}"
else
    rm -rf "$WORK/config"
fi
C="$WORK/config"
INC="$C/includes.chroot"
mkdir -p "$C/package-lists" "$C/hooks/normal" "$INC"

# ---------------------------------------------------------------- packages
cat > "$C/package-lists/sentinel.list.chroot" <<'EOF'
live-boot live-config live-config-systemd systemd-sysv
sudo ca-certificates curl git gcc libc6-dev make
nano less htop procps iproute2 iputils-ping usbutils pciutils
locales console-setup keyboard-configuration
network-manager wpasupplicant wireless-regdb
firmware-linux-free firmware-misc-nonfree firmware-iwlwifi firmware-realtek
firmware-atheros firmware-brcm80211 firmware-amd-graphics
intel-microcode amd64-microcode
EOF
if [ "$LITE" = 0 ]; then
    cat > "$C/package-lists/desktop.list.chroot" <<'EOF'
xorg xfce4 xfce4-terminal lightdm lightdm-gtk-greeter dbus-x11 polkitd
firefox-esr network-manager-gnome thunar mousepad xdg-utils fonts-dejavu-core
EOF
fi

# ---------------------------------------------------------------- Sentinel itself
S="$INC/opt/sentinel"
mkdir -p "$S/src" "$S/bin"
for f in main.c export_model.c Makefile fetch_corpus.sh start.sh demo.sh cluster.sh \
         recursive-learn.sh nodes.example README.md LICENSE CHANGELOG.md CLUSTER.md \
         RELEASE_NOTES.md PRO.md; do
    [ -e "$REPO/$f" ] && cp "$REPO/$f" "$S/src/"
done
cp -r "$REPO/assets" "$S/src/"
cp "$HERE/models.tsv" "$S/models.tsv"
cp "$REPO/LICENSE" "$S/LICENSE"
if [ -n "$MODELS" ]; then
    mkdir -p "$S/models"
    n=0; for f in "$MODELS"/*.bin; do [ -f "$f" ] && cp "$f" "$S/models/" && n=$((n+1)); done
    say "baked $n trained checkpoint(s) into the image"
fi
[ -n "$CORPUS" ] && cp -r "$CORPUS" "$S/corpus" && say "baked corpus ($(du -sh "$CORPUS" | cut -f1))"

install -D -m 0755 "$HERE/sentinel-start"     "$INC/usr/local/bin/sentinel-start"
install -D -m 0755 "$HERE/sentinel-wallpaper" "$INC/usr/local/bin/sentinel-wallpaper"
install -D -m 0755 "$HERE/hook-sentinel.sh"   "$C/hooks/normal/9000-sentinel.hook.chroot"

# ---------------------------------------------------------------- branding
say "rendering branding from $MARK"
mark_png="$WORK/mark-600.png"
rsvg-convert -h 600 "$REPO/$MARK" -o "$mark_png"
mark_b64="$(base64 -w0 "$mark_png")"
cat > "$WORK/wallpaper.svg" <<EOF
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1920" height="1080">
  <defs><radialGradient id="g" cx="50%" cy="38%" r="75%">
    <stop offset="0" stop-color="#14233a"/><stop offset="1" stop-color="$BG_COLOR"/></radialGradient></defs>
  <rect width="1920" height="1080" fill="url(#g)"/>
  <image x="821" y="190" width="278" height="300" xlink:href="data:image/png;base64,$mark_b64"/>
  <text x="960" y="620" text-anchor="middle" font-family="DejaVu Sans Mono, monospace" font-weight="bold"
        font-size="96" letter-spacing="6" fill="#e6f6ff">$OS_NAME</text>
  <text x="960" y="700" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="34" fill="$ACCENT">$TAGLINE</text>
</svg>
EOF
mkdir -p "$INC/usr/share/backgrounds"
rsvg-convert "$WORK/wallpaper.svg" -o "$INC/usr/share/backgrounds/sentinel-os.png"
cat > "$WORK/icon.svg" <<EOF
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="256" height="256">
  <rect width="256" height="256" rx="48" fill="$BG_COLOR"/>
  <image x="44" y="24" width="168" height="182" xlink:href="data:image/png;base64,$mark_b64"/>
  <text x="128" y="236" text-anchor="middle" font-family="DejaVu Sans Mono, monospace" font-weight="bold" font-size="22" fill="#e6f6ff">SENTINEL</text>
</svg>
EOF
rsvg-convert "$WORK/icon.svg" -o "$S/sentinel.png"

mkdir -p "$INC/etc"
printf '%s %s (based on Debian %s)\n%s\n\n' "$OS_NAME" "$VERSION" "$DEBIAN_RELEASE" "$TAGLINE" > "$INC/etc/issue.sentinel"
cat > "$INC/etc/sentinel-os-release" <<EOF
NAME="$OS_NAME"
VERSION="$VERSION"
EDITION="$EDITION"
BASED_ON="Debian $DEBIAN_RELEASE"
PUBLISHER="$PUBLISHER"
EOF
cat > "$C/hooks/normal/9100-branding.hook.chroot" <<'EOF'
#!/bin/sh
# show the Sentinel banner above the Debian one on the console login
[ -f /etc/issue.sentinel ] && cat /etc/issue.sentinel /etc/issue > /etc/issue.new && mv /etc/issue.new /etc/issue
rm -f /etc/issue.sentinel
EOF
chmod 0755 "$C/hooks/normal/9100-branding.hook.chroot"

mkdir -p "$INC/usr/share/doc/sentinel-os"
cat > "$INC/usr/share/doc/sentinel-os/SOURCES.txt" <<EOF
$OS_NAME is built from Debian $DEBIAN_RELEASE packages plus Sentinel ($VERSION,
Apache-2.0, source in /opt/sentinel/src). Source code for every Debian package in
this image is available from https://sources.debian.org and
https://snapshot.debian.org, and on request from the publisher ($PUBLISHER).
EOF

# boot menu: replace live-build's generic "Live system" label with the OS name
if [ -d /usr/share/live/build/bootloaders ]; then
    cp -r /usr/share/live/build/bootloaders "$C/"
    { grep -rl "Live system" "$C/bootloaders" 2>/dev/null || true; } | while read -r f; do
        sed -i "s/Live system/$OS_NAME/g" "$f"
    done
fi

# ---------------------------------------------------------------- desktop vs lite UX
apps="$INC/usr/share/applications"; mkdir -p "$apps"
desktop_entry() {   # file name exec comment terminal
    cat > "$apps/$1" <<EOF
[Desktop Entry]
Type=Application
Name=$2
Comment=$4
Exec=$3
Icon=/opt/sentinel/sentinel.png
Terminal=$5
Categories=Security;Utility;
EOF
}
desktop_entry sentinel-gui.desktop      "Sentinel"                 "sentinel-start --gui"       "Local AI security agent (web GUI)" false
desktop_entry sentinel-tui.desktop      "Sentinel Terminal"        "sentinel-start --tui"       "Sentinel terminal UI" true
desktop_entry sentinel-loop.desktop     "Sentinel Chat + Learn"    "sentinel-start --loop"      "Interactive loop: auto sub-agents + online learning" true
desktop_entry sentinel-learn.desktop    "Sentinel: Train Model"    "sentinel-start --learn"     "Train the biggest model that fits this PC" true
desktop_entry sentinel-which.desktop    "Sentinel: Which Model?"   "sh -c 'sentinel-start --which; read -r _'" "Show the model picked for this PC" true

if [ "$LITE" = 0 ]; then
    mkdir -p "$INC/etc/xdg/autostart"
    cp "$apps/sentinel-gui.desktop" "$INC/etc/xdg/autostart/sentinel.desktop"
    cat > "$INC/etc/xdg/autostart/sentinel-wallpaper.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Sentinel wallpaper
Exec=sentinel-wallpaper
OnlyShowIn=XFCE;
NoDisplay=true
EOF
    mkdir -p "$INC/etc/lightdm/lightdm-gtk-greeter.conf.d"
    printf '[greeter]\nbackground=/usr/share/backgrounds/sentinel-os.png\n' \
        > "$INC/etc/lightdm/lightdm-gtk-greeter.conf.d/50-sentinel.conf"
else
    mkdir -p "$INC/etc/profile.d"
    cat > "$INC/etc/profile.d/sentinel-tui.sh" <<'EOF'
# Lite edition: open the Sentinel TUI on the first console after auto-login.
if [ "$(tty)" = "/dev/tty1" ] && [ -z "${SENTINEL_TUI_STARTED:-}" ] && command -v sentinel-start >/dev/null 2>&1; then
    export SENTINEL_TUI_STARTED=1
    sentinel-start --tui
    echo "Sentinel exited. Type 'sentinel-start --tui' to reopen, 'sentinel-start --help' for more."
fi
EOF
fi

if [ "$STAGE_ONLY" = 1 ]; then
    say "stage-only: build tree ready at $WORK (live-build not run)"
    exit 0
fi

# ---------------------------------------------------------------- build
say "running live-build (20–60 min; downloads ~1 GB of Debian packages)"
lb build 2>&1 | tee "$WORK/build.log"
iso="$(ls -t "$WORK"/*.iso 2>/dev/null | head -1)"
[ -n "$iso" ] || die "live-build finished without an ISO — see $WORK/build.log"
cp "$iso" "$OUT/$IMAGE_NAME.iso"
( cd "$OUT" && sha256sum "$IMAGE_NAME.iso" > "$IMAGE_NAME.iso.sha256" )
if [ "$WITH_SOURCE" = 1 ]; then
    for t in "$WORK"/*source*.tar*; do [ -f "$t" ] && cp "$t" "$OUT/"; done
fi
say "done: $OUT/$IMAGE_NAME.iso ($(du -h "$OUT/$IMAGE_NAME.iso" | cut -f1))"
echo "    write it to a stick with:  sudo ./usb/make-usb.sh $OUT/$IMAGE_NAME.iso /dev/sdX"
