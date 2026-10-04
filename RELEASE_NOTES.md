# Release notes

## Sentinel v0.6.0

A self-contained AI security agent in one C file — now with optional LayerNorm, a
gradient checker, weight tying, and quantization for small devices.

### New since v0.3.x

- **LayerNorm GRU** (`make ln` / `-DUSE_LN`) — converges faster (3.28 vs 3.75
  loss/char at the same budget). Gradient-checked.
- **Gradient checker** (`make gradcheck` / `--gradcheck`) — verifies every gradient
  against finite differences.
- **Weight tying** (`-DTIE_WEIGHTS`) — reuse the embedding as the output layer
  (needs `EMBED == HIDDEN`).
- **Safer long training** — LR warmup, auto-checkpoint every 2000 iterations,
  `SLM_LR` override.
- **Quantization** — `--quantize int8|1bit` exports small models; `--quantize-train`
  (QAT) keeps 1-bit models coherent.
- **On-device models** — `make pet-model` trains a tiny 1-bit model and emits
  `model.h` for microcontrollers.
- **8 GB Raspberry Pi 5 flagship** — `make huge` (~143M params).
- **Opt-in online advisor** — `--ask-opus` / `--self-study` (needs
  `ANTHROPIC_API_KEY`; fully local otherwise).

All new training options are off by default; the standard build is unchanged.
Per-version details (v0.4.0 – v0.6.0) are in [CHANGELOG.md](CHANGELOG.md).

### Quick start

```sh
git clone https://github.com/Mattmorris-dev/Sentinel-SLM && cd Sentinel-SLM
make && ./start.sh
```
