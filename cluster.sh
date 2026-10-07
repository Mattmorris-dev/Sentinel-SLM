#!/usr/bin/env bash
# Sentinel cluster dispatcher — spread sub-agents across machines over SSH.
#
# A Pi cluster (or any mix of Linux/macOS boxes: Pis + a mini-PC + a laptop)
# doesn't make ONE model bigger, but it does run far more agents at once. This
# fans out Sentinel's `--agent` workers across every node you list and collects
# the output. Each node runs the SAME safety guard, RX-only, as a local agent.
#
# Nodes live in a `nodes` file (override with NODES_FILE=...), one per line:
#     user@host                      # uses ~/sentinel on that host
#     user@host:/opt/sentinel        # or an explicit working dir
#     # lines starting with # are comments
#
# Usage:
#   ./cluster.sh list                 check which nodes are up + have sentinel
#   ./cluster.sh deploy               copy the local ./sentinel to every node
#   ./cluster.sh run "<task>" [N]     run N agents (default 1 per node) spread
#                                     round-robin across the nodes, concurrently
#   ./cluster.sh fanout N "<task>"    same as run with the count first
#
# Requires: ssh/scp to each node with key auth, and a `sentinel` binary there
# (use `deploy`). Pure bash — runs from any node as the controller.
set -uo pipefail

NODES_FILE="${NODES_FILE:-nodes}"
SSH_OPTS="${SSH_OPTS:--o ConnectTimeout=8 -o BatchMode=yes}"

die() { echo "cluster: $*" >&2; exit 1; }
[ -f "$NODES_FILE" ] || die "no nodes file '$NODES_FILE' (copy nodes.example -> nodes)"

# Parse nodes file into parallel arrays: HOSTS[] (user@host) and DIRS[] (workdir)
HOSTS=(); DIRS=()
while IFS= read -r line; do
    line="${line%%#*}"; line="$(echo "$line" | xargs)"   # strip comment + trim
    [ -z "$line" ] && continue
    case "$line" in
        *:*) HOSTS+=("${line%%:*}"); DIRS+=("${line#*:}") ;;
        *)   HOSTS+=("$line");       DIRS+=('~/sentinel') ;;
    esac
done < "$NODES_FILE"
N=${#HOSTS[@]}
[ "$N" -gt 0 ] || die "no nodes listed in '$NODES_FILE'"

cmd_list() {
    for i in $(seq 0 $((N-1))); do
        h="${HOSTS[$i]}"; d="${DIRS[$i]}"
        if out=$(ssh $SSH_OPTS "$h" "cd $d 2>/dev/null && ./sentinel --version" 2>/dev/null); then
            echo "  [up]   $h ($d) — $out"
        else
            echo "  [DOWN] $h ($d) — unreachable or no ./sentinel"
        fi
    done
}

cmd_deploy() {
    [ -x ./sentinel ] || die "no local ./sentinel to deploy (run 'make' first)"
    for i in $(seq 0 $((N-1))); do
        h="${HOSTS[$i]}"; d="${DIRS[$i]}"
        echo "  -> $h:$d"
        ssh $SSH_OPTS "$h" "mkdir -p $d" 2>/dev/null \
            && scp -q ./sentinel "$h:$d/sentinel" \
            && ssh $SSH_OPTS "$h" "chmod +x $d/sentinel" \
            && echo "     ok" || echo "     FAILED"
    done
}

# Run one agent for a base64-encoded task on node index $1; label output.
run_one() {
    local i="$1" tag="$2" b64="$3"
    local h="${HOSTS[$i]}" d="${DIRS[$i]}"
    local rc="cd $d && ./sentinel --agent \"\$(printf %s '$b64' | base64 -d)\""
    { echo "===== agent $tag @ $h ====="; ssh $SSH_OPTS "$h" "$rc" 2>&1; }
}

cmd_run() {
    local task="$1" count="${2:-$N}"
    [ -n "$task" ] || die "empty task"
    local b64; b64="$(printf %s "$task" | base64 | tr -d '\n')"
    echo "cluster: dispatching $count agent(s) across $N node(s) for: $task"
    local pids=() tmp; tmp="$(mktemp -d)"
    for k in $(seq 0 $((count-1))); do
        i=$(( k % N ))
        run_one "$i" "$((k+1))/$count" "$b64" > "$tmp/$k.out" 2>&1 &
        pids+=("$!")
    done
    for p in "${pids[@]}"; do wait "$p"; done
    for k in $(seq 0 $((count-1))); do cat "$tmp/$k.out"; echo; done
    rm -rf "$tmp"
    echo "cluster: $count agent(s) done."
}

case "${1:-}" in
    list)   cmd_list ;;
    deploy) cmd_deploy ;;
    run)    shift; cmd_run "${1:-}" "${2:-}" ;;
    fanout) shift; cmd_run "${2:-}" "${1:-}" ;;   # fanout N "task"
    *) echo "usage: $0 {list|deploy|run \"<task>\" [N]|fanout N \"<task>\"}"; exit 1 ;;
esac
