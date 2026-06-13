# Sentinel — self-contained deep RNN agent in pure C.
# Free software (MIT). No external ML libraries; only libc + libm.

CC      ?= gcc
CFLAGS  ?= -O2 -Wall
LDLIBS   = -lm
BIN      = sentinel
PREFIX  ?= /usr/local
VERSION ?= 0.1.0

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
	    main.c fetch_corpus.sh Makefile README.md LICENSE .gitignore
	@echo "built sentinel-$(VERSION)-src.tar.gz"

help:                  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
