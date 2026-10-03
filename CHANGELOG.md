# Changelog

All notable changes to Sentinel. Each section below is ready to copy-paste into a
GitHub Release. Format loosely follows [Keep a Changelog](https://keepachangelog.com).

---

## v0.6.0

**LayerNorm.**

- `-DUSE_LN` / `make ln`: optional Layer-Normalized GRU (learnable gain per gate
  channel, with the existing bias as the post-norm shift). Converges faster — e.g.
  3.28 vs 3.75 loss/char at the same budget. Verified by `--gradcheck`. LN checkpoints
  use a new format tag (`SLM4`) so they can't be cross-loaded with non-LN builds.
- Off by default; the standard build is unchanged.

## v0.5.2

**Gradient checker and weight tying.**

- `--gradcheck` / `make gradcheck`: numerically verifies every analytic gradient against
  finite differences (build tiny first). The safety net for any change to the network
  math; the default build passes at ~1e-6.
- `-DTIE_WEIGHTS`: optional weight tying — reuse the input embedding as the output
  projection, so the embedding is trained by both paths. Requires `EMBED == HIDDEN`
  (compile-time checked); no new arrays, no checkpoint-format change. Gradient-checked.

## v0.5.1

**Safer long training runs.**

- Auto-checkpoint every 2000 iterations of a long `--train` / `--quantize-train` run
  (`-DCKPT_EVERY=N` to change), so an interrupted run resumes instead of starting over.
- `SLM_LR` environment variable overrides the base learning rate at runtime.

## v0.5.0

**On-device models, QAT, an opt-in advisor, and the 8 GB flagship.**

- `--quantize-train int8|1bit`: quantization-aware training (straight-through
  estimator) — turns 1-bit models from gibberish into coherent text.
- `make pet-model` + `export_model.c`: one command trains the tiny on-device model with
  1-bit QAT and emits `model.h` (`const` arrays) for microcontroller firmware.
- `--ask-opus` / `--self-study`: opt-in online advisor (needs `ANTHROPIC_API_KEY`;
  fully local otherwise). Answers are only printed or trained on, never executed.
- `make huge` resized to max out an 8 GB Raspberry Pi 5 (~143M params).
- Linear learning-rate warmup (first 200 steps of a fresh model, then constant, so
  online learning keeps working across runs).

## v0.4.0

**Weight quantization (export).**

- `--quantize int8|1bit` exports a compact copy of the trained model — `int8` ~8×
  smaller than the weights (essentially lossless), `1bit` ~64× smaller (sign + per-row
  magnitude) — **without** touching the float checkpoint (it refuses to overwrite it).
- `--sample-quant <file>` loads a quantized model and samples from it to verify.
- `make quant` writes both exports in one step.

## v0.3.1

**Open-core positioning, landing page, and polish.**

- Open-core model: `PRO.md` (commercial offering) + a Free-vs-Pro table; README and
  PITCH explain the split (Core is free/Apache-2.0, Pro is convenience + extras).
- README now leads with the product pitch, then flows into the docs; `PITCH.md` is the
  standalone one-pager.
- `demo.sh` showcase + a "See it run" section with real captured output.
- GitHub badges (license, language, release, stars/forks/issues, "no dependencies",
  "100% local").
- `assets/tui.svg` + `assets/gui.svg` interface mockups.

## v0.3.0

**Adam optimizer, model tiers, retrieval over CVE/malware/OSINT, GUI + TUI.**

- **Adam optimizer** (m + v moments, bias correction) replaces Adagrad; persisted in the
  checkpoint. The model now produces coherent text instead of character soup.
- **Three build tiers from one source** (size knobs are `-D`-overridable):
  `make slm` (~1.2M params, 27 MB) · `make` (~58M, ~1.9 GB, **plain gcc**) ·
  `make huge` (~101M, ~3.2 GB, `-mcmodel=large`).
- **Retrieval (RAG)** generalized to **CVE / malware / OSINT** topics; greps a local
  corpus, with **live internet** CVE-id lookup via `SLM_ONLINE=1`. `fetch_corpus.sh` adds
  malware + OSINT sources.
- **Web GUI** (`--serve`, localhost-only HTTP server, pure C) and **TUI** (`--tui`).
- Cleaner retrieval output (skips badge/blank lines).

## v0.2.0

**Deeper/bigger network, real-world data, fan-out, hardening, packaging.**

- Stacked multi-layer network with a learned embedding; scaled-up parameters.
- **Data consolidation:** `--train` accepts files *and* directories (recursively folded
  into one corpus). `fetch_corpus.sh` pulls high-starred security/coding repos + CVE data.
- **Difficulty-tiered sub-agent fan-out:** `FANOUT: <n>` spawns up to 100 concurrent
  agents, each sized to task difficulty (light/medium/heavy).
- **CVE lookup agent** that greps the corpus for real vulnerability writeups.
- **Security hardening:** non-truncating command guard, expanded denylist, unique
  `mkdtemp` temp dirs for the coding agent, shell-input validation.
- Apache-2.0 license, `Makefile`, `start.sh` launcher, built-in `--help`, SVG logo.

## v0.1.0

**Initial release — a self-contained Small Language Model in pure C.**

- Character-level tokenizer, static weight matrices, hand-written matrix multiply.
- Recurrent network with hidden-state feedback (no fixed token-context window),
  trained by backprop-through-time; checkpoint persistence + online learning.
- Sub-agent layer: a `TASK:` trigger `fork()`/`exec()`s a separate process; hard safety
  guard refuses `sudo` and destructive commands; `SLM_NO_EXEC` plan-only mode.
- No external machine-learning libraries — libc + libm only.
