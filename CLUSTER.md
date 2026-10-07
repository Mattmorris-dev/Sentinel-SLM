# Cluster & recursive learning

Two ways to get "more" out of Sentinel without a bigger single box: fan agents
out across several machines, and let the model keep practicing on itself.

## Cluster: fan agents out across machines (`cluster.sh`)

A cluster does **not** make one model bigger (see "Why a cluster can't grow one
model" below) — but it runs many more agents at once. `cluster.sh` spreads
Sentinel's `--agent` workers across every machine you list and collects the
output. Any mix works: Raspberry Pis, a mini-PC, a laptop. Each node runs the
same RX-only safety guard a local agent does.

```sh
cp nodes.example nodes        # then edit: one `user@host[:/dir]` per line
make                          # build ./sentinel locally
./cluster.sh deploy           # copy the binary to every node
./cluster.sh list             # check which nodes are up + have sentinel
./cluster.sh run "show me a cve about http" 12   # 12 agents, spread round-robin
./cluster.sh fanout 50 "scan local listening ports"
```

Requirements: key-based `ssh`/`scp` to each node, and a `sentinel` binary there
(`deploy` handles it). The task is base64-encoded over SSH, so quoting and shell
injection are non-issues. Your `nodes` file is gitignored (it names your hosts).

Runs as the orchestrator from any machine — including one of the nodes.

## Recursive self-learning (`recursive-learn.sh`)

Each round the model **generates** text from its checkpoint, then **retrains on
that text mixed with the real corpus**, and checkpoints. It's a reinforcement
loop that sharpens what the model already knows.

```sh
./recursive-learn.sh                              # 5 rounds on ./corpus
ROUNDS=10 SLM_EPOCHS=4000 CORPUS=./my_data ./recursive-learn.sh
```

**The corpus anchor is not optional.** Training a model purely on its own output
causes *model collapse* — it spirals into gibberish within a few rounds. The real
corpus must dominate every round, which is why the loop always retrains on it
alongside the self-generated text. This loop reinforces; it doesn't create
knowledge that isn't in the corpus. The underlying `--generate <n>` flag (pure
inference from the checkpoint, no training) is useful on its own.

## Why a cluster can't grow one model

Sentinel is a single process that holds all weights in one machine's RAM with a
hand-written matmul. Nothing in it shards across nodes. To run one model across
several Pis you'd have to split the weight matrices and ship the hidden state
between nodes **every timestep** — and a Pi's ~1 Gb/s network is far slower than
its RAM, so the result would be *slower* than one Pi, not faster. RNN/BPTT is a
latency-bound sequential loop; clusters win on big chunky parallel work, not this.

**For a bigger single model, add RAM to one box:** a 16 GB Raspberry Pi 5 gets you
~400–500M params; a mini-PC or workstation with 32–64 GB goes much further. Just
raise `-DHIDDEN` and rebuild — the code is dimension-generic (`make max` already
pushes an 8 GB Pi 5 to ~185M).

## About a GPU (and the Pi NVMe/PCIe question)

Two independent facts worth being clear on:

1. **Sentinel is pure C / CPU — there is no GPU code path.** Plugging in a GPU
   does nothing for it today; using one would require writing a CUDA (or OpenCL)
   matmul and a device memory path, which is a separate project, not a cable.
2. **A GPU on a Raspberry Pi 5 is impractical.** The Pi 5 exposes a single
   PCIe 2.0 **x1** lane on its FPC connector. NVMe HATs use that one lane; running
   an NVMe **and** a GPU means putting a **PCIe switch/bridge** on that lane (what
   a "dual" PCIe HAT does) and splitting already-tiny bandwidth, and desktop GPUs
   mostly don't even enumerate/driver-load on ARM. It's a hobbyist curiosity, not
   a path to performance.

If you want GPU acceleration, host the GPU on the **mini-PC or laptop** (a real
PCIe x16 slot, or an eGPU over Thunderbolt/OCuLink) and run the GPU build there —
once that build exists. For storage, a single NVMe HAT on the Pi's one lane is the
sensible use of it; keep the GPU off the Pi.
