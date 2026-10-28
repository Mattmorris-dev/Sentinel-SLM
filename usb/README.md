# Sentinel OS — the all-in-one bootable USB

One USB stick that boots any x86-64 PC (BIOS or UEFI) into Debian 13 with every
Sentinel model and agent built in. On boot, `sentinel-start` checks how much RAM
the computer has and runs the **biggest model that fits**:

| model | params | RAM to train | picked when the PC has about… |
|-------|--------|--------------|-------------------------------|
| slm   | ~0.8M  | 40 MB        | anything (always the fallback) |
| std   | ~58M   | 1.9 GB       | 4 GB |
| huge  | ~143M  | 4.6 GB       | 6 GB |
| max   | ~185M  | 5.9 GB       | 8 GB |
| xl    | ~383M  | 12.3 GB      | 16 GB |
| xxl   | ~738M  | 23.6 GB      | 32 GB |

Trained checkpoints on the stick are used before untrained ones. Older
checkpoints still load, because the launcher reads each file's header and
compiles a matching binary for it.

There are two editions:

- **desktop** (default, about 2 GB): an XFCE desktop that opens the Sentinel web GUI in the browser.
- **lite** (`--lite`, about 0.8 GB): no desktop. It boots straight into the Sentinel terminal UI.

## 1. Build the image

This needs an x86-64 machine with internet access.

```sh
sudo ./usb/build-iso.sh            # on Debian 13
./usb/build-in-docker.sh           # anywhere with Docker (Linux, WSL2, Intel Mac)
./usb/build-in-docker.sh --lite
```

The result is written to `usb/build/out/sentinel-os-<version>-<edition>.iso`.

To build in the cloud instead, move `usb/github-workflow.yml` to
`.github/workflows/build-usb.yml` and push it (one time). Then open the
repo's **Actions** tab, pick
**Build Sentinel OS USB image**, and click **Run workflow** (pushing a `v*` tag
also starts it). Both ISOs come back as downloadable artifacts, kept for 14 days.

## 2. Write a stick

```sh
sudo ./usb/make-usb.sh usb/build/out/sentinel-os-*-desktop.iso /dev/sdX \
     --models ~/my-trained-models --corpus ~/my-corpus
```

This erases the stick, writes the image, and adds a `persistence` partition, so
learning carries over between reboots. It then copies your trained `.bin` files
and corpus onto the stick. It refuses to touch the disk your system runs from
and any disk that isn't removable (unless you pass `--force`), and it makes you
type the device name before it writes. Use `--dry-run` to preview what it would do.

### Which stick size?

The image itself is small. What fills a stick is trained checkpoints (they
hold the weights plus optimizer state, about 12 bytes per parameter) and the
corpus. The persistence partition takes all the space left after the image.

| stick | holds |
|-------|-------|
| 8 GB  | the image plus trained slm/std/huge/max checkpoints (~4.6 GB together) |
| 32 GB | adds a trained **xl** (~4.4 GB) and a large corpus |
| 64 GB | adds a trained **xxl** (~8.5 GB) too, so every tier is on one stick with plenty of room to keep learning |

A bigger stick doesn't make a model smarter on its own. It lets you carry
bigger trained models, and the PC's RAM still decides which one runs. Train
xl and xxl on a 16 GB / 32 GB machine first, then load them with `--models`.

If you only need a stick that boots, without models or persistence, any image
writer works (Balena Etcher, Rufus in DD mode, `dd`).

## 3. Rebrand it

Edit `usb/brand.conf` and rebuild. You can change the OS name, hostname/user,
publisher, ISO label, tagline, colours, and Debian release. To use your own logo,
replace `usb/brand-mark.svg` (or point `MARK=` at another SVG). The wallpaper,
app icon and boot menu are all generated from it.

## 4. Selling it

- **Keep the moat off the public image.** Trained models and the corpus never go
  into the repo or the free ISO. Load them per stick with `make-usb.sh --models`
  (that is the paid edition), or bake a private build with
  `build-iso.sh --models DIR` and never upload it.
- **GPL compliance.** The image is built from Debian, which is free software.
  Selling it is allowed, but you must offer the source. Every image ships
  `/usr/share/doc/sentinel-os/SOURCES.txt` saying where to get it. Build with
  `--with-source` to also produce the matching source tarball to hand out on
  request.
- **Trademarks.** Don't call the product "Debian", and don't use the Debian
  logo as your brand. "Built on Debian" is fine.

## Files

| file | what it does |
|------|--------------|
| `build-iso.sh` | builds the ISO with live-build (`--stage-only` prepares the tree without building) |
| `build-in-docker.sh` | runs `build-iso.sh` inside a `debian:trixie` container |
| `make-usb.sh` | writes a stick, adds persistence, and loads models and corpus |
| `brand.conf`, `brand-mark.svg` | branding |
| `models.tsv` | the model lineup: binary name, checkpoint, RAM needed, compile flags |
| `hook-sentinel.sh` | compiles every model inside the image at build time |
| `sentinel-start` | launcher on the stick: RAM-aware model pick, imports checkpoints, modes `--gui/--tui/--loop/--learn/--recursive/--generate/--which` |
| `sentinel-wallpaper` | applies the branded wallpaper on every monitor |
| `github-workflow.yml` | cloud build (free editions only, never add trained models) |
