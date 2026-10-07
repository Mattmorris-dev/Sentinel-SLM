# Sentinel — self-contained deep GRU agent in pure C.
# Free software (Apache-2.0). No external ML libraries; only libc + libm.

CC      ?= gcc
# The default ~50M-param model uses <2 GB of static arrays, so it builds with
# plain gcc and runs anywhere. (The 'huge' target adds -mcmodel=large for the
# 101M model, whose arrays exceed the 2 GB small-code-model limit.)
# LayerNorm (-DUSE_LN) is ON by default: it's gradient-checked and converges to a
# lower loss at the same budget, so the standard build is the stronger model.
# Override with `make USE_LN=0` for the plain GRU.
USE_LN  ?= 1
LNFLAG   = $(if $(filter 0,$(USE_LN)),,-DUSE_LN)
CFLAGS  ?= -O2 -Wall $(LNFLAG)
LDLIBS   = -lm
BIN      = sentinel
PREFIX  ?= /usr/local
VERSION ?= 0.6.0

.PHONY: all both slm huge max ln gradcheck run train quant pet-model fetch quick-fetch clean distclean install uninstall package help

all: $(BIN)            ## build the default ~50M model (plain gcc, runs anywhere)

$(BIN): main.c
	$(CC) $(CFLAGS) -o $@ main.c $(LDLIBS)

# A tiny SLM from the SAME source (~1.2M params, 27 MB). Runs on anything.
slm: main.c            ## build a tiny SLM (sentinel-slm, ~1.2M params)
	$(CC) -O2 -Wall $(LNFLAG) -DHIDDEN=256 -DEMBED=64 -DNUM_LAYERS=2 \
	    -DCKPT_PATH='"sentinel-slm.bin"' -o sentinel-slm main.c $(LDLIBS)

# The flagship model, sized to MAX OUT an 8 GB Raspberry Pi 5 (the biggest Pi):
# ~142M params, ~4.6 GB to train (Adam ~32 B/param), leaving headroom for the OS.
# Needs the large code model (>2 GB static arrays). Train on a fast box if you
# can — a real train at this width takes many CPU-hours — then ship an int8
# export (`make quant`) to the Pi for fast, low-RAM inference.
huge: main.c           ## build the 8 GB Pi 5 flagship (sentinel-huge, ~142M params)
	$(CC) -O2 -Wall $(LNFLAG) -mcmodel=large -DHIDDEN=2816 \
	    -DCKPT_PATH='"sentinel-huge.bin"' -o sentinel-huge main.c $(LDLIBS)

# Pushes an 8 GB Pi 5 to its limit: ~185M params, ~5.9 GB to train. Leaves only
# ~1-1.5 GB headroom, so close other apps; train on a bigger box if you can.
max: main.c            ## push an 8 GB Pi 5 (~185M params, ~5.9 GB train)
	$(CC) -O2 -Wall $(LNFLAG) -mcmodel=large -DHIDDEN=3200 \
	    -DCKPT_PATH='"sentinel-max.bin"' -o sentinel-max main.c $(LDLIBS)

ln: main.c             ## build with LayerNorm (sentinel-ln, faster convergence)
	$(CC) -O2 -Wall -DUSE_LN -DHIDDEN=256 -DEMBED=64 -DNUM_LAYERS=2 \
	    -DCKPT_PATH='"sentinel-ln.bin"' -o sentinel-ln main.c $(LDLIBS)

gradcheck: main.c      ## numerically verify backprop (tiny build + --gradcheck)
	$(CC) -O2 -Wall -DHIDDEN=16 -DEMBED=8 -DNUM_LAYERS=2 \
	    -DCKPT_PATH='"/tmp/gc.bin"' -o sentinel-gradcheck main.c $(LDLIBS)
	./sentinel-gradcheck --gradcheck

both: $(BIN) slm       ## build the default model + the tiny SLM

run: $(BIN)            ## train on the built-in corpus, then serve the read/agent loop
	./$(BIN)

train: $(BIN)          ## train on ./corpus (run `make fetch` first)
	./$(BIN) --train corpus

quant: $(BIN)          ## export small quantized models (int8 ~8x, 1bit ~64x) — float ckpt untouched
	./$(BIN) --quantize int8 && ./$(BIN) --quantize 1bit

# One command to produce the on-device ESP32 model: train a tiny net
# (HIDDEN=128/EMBED=32/LAYERS=2) with quantization-AWARE 1-bit training, then
# emit model.h for the Sentinel Pet firmware. Override SLM_EPOCHS / PET_CORPUS.
# (tiny + QAT-1bit is the validated on-device config: ~38 KB, stays coherent.)
pet-model: main.c export_model.c   ## train tiny+QAT-1bit and emit model.h for the ESP32 Pet
	$(CC) -O2 -Wall -DHIDDEN=128 -DEMBED=32 -DNUM_LAYERS=2 \
	    -DCKPT_PATH='"sentinel-pet.bin"' -o sentinel-pet-trainer main.c $(LDLIBS)
	SLM_EPOCHS=$${SLM_EPOCHS:-6000} ./sentinel-pet-trainer --quantize-train 1bit $${PET_CORPUS:-}
	$(CC) -O2 -Wall -o export_model export_model.c
	./export_model sentinel-1bit-qat.bin model.h
	@echo "model.h ready (tiny + QAT 1-bit). Copy it into the private sentinel-pet firmware."

fetch:                 ## download the full real-world corpus (security/coding/CVE)
	./fetch_corpus.sh

quick-fetch:           ## download a small/fast demo corpus
	QUICK=1 ./fetch_corpus.sh

clean:                 ## remove the binary and checkpoint
	rm -f $(BIN) sentinel.bin sentinel-int8.bin sentinel-1bit.bin \
	    sentinel-pet-trainer sentinel-pet.bin sentinel-1bit-qat.bin \
	    sentinel-int8-qat.bin export_model model.h \
	    sentinel-ln sentinel-ln.bin sentinel-max sentinel-max.bin sentinel-gradcheck

distclean: clean       ## also remove the fetched corpus
	rm -rf corpus

install: $(BIN)        ## install to $(PREFIX)/bin
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(BIN) $(DESTDIR)$(PREFIX)/bin/$(BIN)

uninstall:             ## remove the installed binary
	rm -f $(DESTDIR)$(PREFIX)/bin/$(BIN)

package:               ## build a source release tarball
	tar czf sentinel-$(VERSION)-src.tar.gz \
	    main.c export_model.c fetch_corpus.sh start.sh demo.sh cluster.sh recursive-learn.sh nodes.example \
	    Makefile README.md PITCH.md PRO.md CHANGELOG.md CLUSTER.md RELEASE_NOTES.md LICENSE .gitignore \
	    assets/sentinel-logo.svg assets/tui.svg assets/gui.svg ci-build.yml
	@echo "built sentinel-$(VERSION)-src.tar.gz"

help:                  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
