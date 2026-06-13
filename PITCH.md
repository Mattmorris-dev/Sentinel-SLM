<p align="center">
  <img src="assets/sentinel-logo.svg" alt="Sentinel" width="560">
</p>

# Sentinel — your security analyst that never leaves the box

**A self-contained AI security agent in one C file. No cloud. No API keys. No vendor. No per-token bill.**
It runs on a $50 Raspberry Pi, answers CVE / malware / OSINT questions from real data,
and spawns its own sub-agents to do the work — entirely on your machine.

```sh
git clone https://github.com/mattymomo1993/Sentinel && cd Sentinel
make && ./start.sh
```

---

## Why it exists

Every other "AI agent" ships your data to someone else's servers and bills you per token.
That's a non-starter for the people who need security tooling the most — **air-gapped
networks, incident responders, privacy-first teams, home labs, classrooms.**

Sentinel flips it: the whole thing is **~3,000 lines of standard C** (libc + libm only,
no PyTorch, no Python, no dependencies). You compile one file and own the binary forever.

## What it does

- 🛡️ **Answers security questions from real data.** Ask *"show me a cve about log4j"* or
  *"malware like emotet"* and it greps a local corpus of **150k+ CVE writeups, malware-analysis
  notes, and OSINT resources** — with optional **live internet lookup** for fresh CVEs.
- 🤖 **Spawns its own sub-agents.** A `TASK:` trigger forks specialist workers
  (cybersecurity / coding / general); `FANOUT: 100` runs up to a hundred at once, each
  sized to the task's difficulty.
- 🧠 **A real neural net, by hand.** A stacked GRU with backprop-through-time and Adam —
  no libraries — that keeps learning from what it reads.
- 🔒 **Safe by default.** Every command is screened (no `sudo`, no destructive ops); the
  web UI binds to localhost only; one env var (`SLM_NO_EXEC`) makes it plan-only.
- 🖥️ **However you like it.** A web GUI (`--serve`), a terminal UI (`--tui`), a plain CLI,
  or a single sub-agent call.

## Runs on what you've got

| Build | Params | RAM | Hardware |
|---|---|---|---|
| `make slm` | ~1.2M | 27 MB | a potato |
| `make` | ~58M | ~1.9 GB | a Pi 4 / any laptop |
| `make huge` | ~101M | ~3.2 GB | an 8 GB Pi / a server |

## Who it's for

Pentesters and blue-teamers who can't send data to the cloud · home-lab and
self-hosting folks · educators teaching how an LLM + agent actually works under the hood ·
anyone who wants an AI tool they fully own.

## Honest bit

Sentinel is a **focused local security assistant**, not a ChatGPT replacement. Its power
is in **retrieval over real data + autonomous sub-agents running locally**, not in
out-writing a 100-billion-parameter cloud model. That's the point: small, private, free,
and yours.

---

**Apache-2.0 licensed.** Free to use, fork, and ship commercially. Built in pure C.
