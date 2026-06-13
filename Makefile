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

.PHONY: all run train fetch quick-fetch clean distclean install uninstall package help

all: $(BIN)            ## build the binary

$(BIN): main.c
	$(CC) $(CFLAGS) -o $@ main.c $(LDLIBS)

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
