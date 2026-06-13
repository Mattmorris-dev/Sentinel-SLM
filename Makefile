# Sentinel — self-contained deep GRU agent in pure C.
# Free software (Apache-2.0). No external ML libraries; only libc + libm.

CC      ?= gcc
# -mcmodel=large is required because the default model uses >2 GB of static
# arrays (it overflows the small code model's 32-bit relocations otherwise).
CFLAGS  ?= -O2 -Wall -mcmodel=large
LDLIBS   = -lm
BIN      = sentinel
PREFIX  ?= /usr/local
VERSION ?= 0.3.0

.PHONY: all both slm run train fetch quick-fetch clean distclean install uninstall package help

all: $(BIN)            ## build the big model (default)

$(BIN): main.c
	$(CC) $(CFLAGS) -o $@ main.c $(LDLIBS)

# A genuine small LM from the SAME source (~1.2M params, no -mcmodel=large).
# Uses its own checkpoint so it can coexist with the big model.
slm: main.c            ## build a small SLM (sentinel-slm, ~1.2M params)
	$(CC) -O2 -Wall -DHIDDEN=256 -DEMBED=64 -DNUM_LAYERS=2 \
	    -DCKPT_PATH='"sentinel-slm.bin"' -o sentinel-slm main.c $(LDLIBS)

both: $(BIN) slm       ## build BOTH the big model and the small SLM

run: $(BIN)            ## train on the built-in corpus, then serve the read/agent loop
	./$(BIN)

train: $(BIN)          ## train on ./corpus (run `make fetch` first)
	./$(BIN) --train corpus

fetch:                 ## download the full real-world corpus (security/coding/CVE)
	./fetch_corpus.sh

quick-fetch:           ## download a small/fast demo corpus
	QUICK=1 ./fetch_corpus.sh

clean:                 ## remove the binary and checkpoint
	rm -f $(BIN) sentinel.bin

distclean: clean       ## also remove the fetched corpus
	rm -rf corpus

install: $(BIN)        ## install to $(PREFIX)/bin
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(BIN) $(DESTDIR)$(PREFIX)/bin/$(BIN)

uninstall:             ## remove the installed binary
	rm -f $(DESTDIR)$(PREFIX)/bin/$(BIN)

package:               ## build a source release tarball
	tar czf sentinel-$(VERSION)-src.tar.gz \
	    main.c fetch_corpus.sh start.sh Makefile README.md LICENSE .gitignore \
	    assets/sentinel-logo.svg .github/workflows/c-cpp.yml
	@echo "built sentinel-$(VERSION)-src.tar.gz"

help:                  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
