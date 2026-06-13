# local_slm — a self-contained deep RNN agent in pure C

A fully self-contained Small Language Model written in standard C with **no
external machine-learning libraries** (only libc + libm). It implements, by hand:

- a **character-level tokenizer** (7-bit ASCII),
- a **learned embedding table**,
- a **stacked, multi-layer recurrent neural network** with hidden-state feedback,
  so it processes a continuous stream with **no fixed token-context window**,
- a **manual matrix-multiply** (`matvec`) used for every layer at every timestep,
- a real **backprop-through-time** trainer (cross-entropy loss, gradient clipping,
  Adagrad),
- **checkpoint persistence** + **online learning** so the network keeps growing its
  skills across runs, and
- a **sub-agent layer**: when it reads a line containing the trigger `TASK:`, it
  `fork()`/`exec()`s a separate process specialised in **cybersecurity / coding /
  general** work, which runs a **safe** shell command and streams the result back.

It's a genuine (if tiny) deep network: real weights, real gradients — just at toy
scale, so it learns character statistics rather than fluent prose. The value is the
complete, dependency-free pipeline.

## Build

```sh
gcc -O2 -Wall -o local_slm main.c -lm
```

## Run

```sh
# Train on the built-in corpus, print the loss curve + a sample, then serve.
# Plain text on stdin -> online learning. Lines with "TASK:" -> spawn a sub-agent.
printf 'TASK: list files in this directory\nTASK: write a C function to reverse a string\n' | ./local_slm
```

Other modes:

```sh
./local_slm --agent "<task>"           # run directly as a single sub-agent
./local_slm --train corpus.txt         # absorb one corpus file, then checkpoint
./local_slm --train ./data             # consolidate a WHOLE directory tree
./local_slm --train a.txt b.md ./src   # consolidate several files + dirs at once
```

### Data consolidation (feed the bigger net more data)

`--train` accepts any mix of files and directories. Directories are walked
recursively and every text/code file (`.txt .md .c .h .py .js .json .csv .html
.java .rs .go .sh …`) is folded into **one** training corpus before training
(dotfiles and `.git` are skipped; total capped at 64 MB). More parameters only
help if they have more data to learn from — this is how you supply it:

```sh
SLM_EPOCHS=20000 ./local_slm --train ./my_notes ./some_repo/src
```

### Fetch real-world data: `fetch_corpus.sh`

To train on real security + coding knowledge and live CVE data, the included
`fetch_corpus.sh` clones a curated set of **high-starred, free, open-source**
GitHub repositories plus public CVE writeups into `./corpus`, then you train on
the whole pile. It is **100% free and vendor-neutral** — only `git` + `curl`,
no API keys, no paid services, no Claude or any other vendor dependency, and the
training runs fully offline afterwards.

```sh
QUICK=1 ./fetch_corpus.sh        # small/fast demo: famous CVEs + a couple repos
./fetch_corpus.sh                # full corpus (several repos + recent CVE years)
CVE_YEARS="2022 2023 2024 2025" ./fetch_corpus.sh
./local_slm --train corpus       # consolidate + train on everything fetched
```

Sources are grouped in editable arrays at the top of the script:
- **Security:** OWASP Cheat Sheets, PayloadsAllTheThings, the-book-of-secret-knowledge
- **Coding:** TheAlgorithms (C and Python)
- **Neural-net reference:** micrograd, char-rnn, nanoGPT (so it can also learn the
  code of models like itself)
- **CVE data:** `trickest/cve` — 150k+ real vulnerability writeups (Log4Shell, xz
  backdoor, Heartbleed, …), pulled per-year via sparse checkout so it stays bounded

Add any repo you like to those lists.

## Architecture (where the depth lives)

```
token --> [embedding] --> [RNN layer 0] --> [RNN layer 1] --> ... --> [softmax] --> next char
                               ^  |              ^  |
                               +--+ hidden state +--+ hidden state   (fed back every step)
```

## Growing the network's skills

- **Deepen / widen it:** edit the hyperparameters at the top of `main.c`
  (`NUM_LAYERS`, `HIDDEN`, `EMBED`, `SEQ_LEN`) and recompile. The layer code is
  dimension-generic, so nothing else needs to change. (Changing the architecture
  starts a fresh checkpoint automatically — the old one records its own dims.)
  The default build is **3 layers, HIDDEN=192, EMBED=64 ≈ 255K parameters**;
  raising `HIDDEN`/`NUM_LAYERS` scales the parameter count roughly with
  `NUM_LAYERS * HIDDEN²`.
- **Teach it more:** `./local_slm --train yourtext.txt`, or just pipe text in on
  stdin during the read loop. Both persist into `local_slm.bin`.
- **Train longer:** `SLM_EPOCHS=20000 ./local_slm`.

## Safety

Sub-agents run real shell commands **autonomously**, but every command is screened
by a hard guard (`is_command_safe`) that **refuses `sudo`** and a denylist of
destructive/privileged patterns (`rm -rf /`, `mkfs`, `dd of=`, fork bombs,
`shutdown`/`reboot`, pipe-to-shell from the network, reads of `/etc/shadow`, …).
Set `SLM_NO_EXEC=1` to force **plan-only** mode (prints the command it *would* run,
executes nothing) as a global kill switch — no recompile needed.
