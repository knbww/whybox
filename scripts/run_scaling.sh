#!/bin/bash
# Drives the scaling grid, one process per training, sized by measured memory.
#
# Peak resident memory per worker, measured on the widest data level:
#   0.043M 0.78 GB   0.263M 1.38 GB   0.585M 1.65 GB   3.20M 3.32 GB   16.29M 8.16 GB
# With ~14 GB to spend, the top rung fits ONE worker; launching eighteen of them
# is what invoked the OOM killer. So concurrency is per capacity, and threads are
# raised as concurrency falls to keep the twenty cores busy. The trainer scales
# poorly with threads (937s at one, 654s at two, 479s at five for the anchor), so
# narrow-and-many is preferred wherever memory allows it.
#
# Resumable: a shard whose output exists is skipped.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/scaling

#            cap0 cap1 cap2 cap3 cap4
WORKERS=(     15    9    7    4    1 )
THREADS=(      1    2    3    5   16 )
CAPS="${CAPS:-3 2 1 0}"          # the 16.29M rung is opt-in: CAPS="4 3 2 1 0"

par () {  # $1 = workers, $2 = threads
  MINT_THREADS="$2" xargs -P "$1" -L1 bash -c \
    'L=$(IFS=-; echo "$*"); .venv/bin/python -u -m scripts.scaling_study "$@" \
       > "results/scaling/log-$L.txt" 2>&1 || echo "FAILED $*"' _
}

[ -f results/scaling/baseline.json ] || MINT_THREADS=6 \
  .venv/bin/python -u -m scripts.scaling_study baseline

for ci in $CAPS; do
  W=${WORKERS[$ci]}; T=${THREADS[$ci]}
  echo "=== capacity $ci: $W workers x $T threads ($(date +%H:%M)) ==="
  for src in interact witness; do for li in 0 1 2; do
    [ -f "results/scaling/lrscan-$src-$ci-$li.json" ] || echo "lr $src $ci $li"
  done; done | par "$W" "$T"
  .venv/bin/python -u -m scripts.scaling_study lrpick > /dev/null
  for di in 3 2 1 0; do for src in interact witness; do for sd in 0 1 2; do
    [ -f "results/scaling/cell-$src-$ci-$di-$sd.json" ] || echo "shard $src $ci $di $sd"
  done; done; done | par "$W" "$T"
  echo "=== capacity $ci done ($(date +%H:%M)), cells: $(ls results/scaling/cell-*.json 2>/dev/null | wc -l)/120 ==="
done

.venv/bin/python -u -m scripts.scaling_study merge
echo "=== finished $(date +%H:%M) ==="
