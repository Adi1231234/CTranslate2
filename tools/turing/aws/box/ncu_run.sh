#!/bin/bash
# Nsight Compute on the production runner (an entry.sh "script" line, privileged job): for each heavy kernel, one
# runner process on the first units of a list with that kernel's launches past the first <skip> measured (time,
# DRAM, L2 and SM throughput, occupancy, where the warps stall). One process per kernel, so each kernel gets its own
# sample of the steady state (one filter for all would sample mostly the most frequent one). Nsight Compute is
# installed here from NVIDIA's apt repository (the image carries only Nsight Systems). The reports and a CSV of the
# metrics stay in this folder.
# usage: ncu_run.sh <package> <runner dir> <units list> <n units> <steps to skip> <launches> [VAR=value ...]
set -uo pipefail
PKG=$1 RUNNER=$2 LIST=$3 N=$4 SKIP=$5 COUNT=$6; shift 6
B=${B:-/opt/wb}
for p in "$PKG" "$RUNNER"; do
  [ -d "$B/$p" ] || aws s3 cp --only-show-errors "$S3/$p.tgz" - | tar xz -C "$B"
done
apt-get update -qq > /dev/null 2>&1
# 2025.3: the CUDA 13.0 generation, as the hosts' driver 580 (a newer one may need a newer driver)
NCU_PKG=$(apt-cache search --names-only '^nsight-compute-2025\.3[0-9.]*$' | awk '{print $1}' | sort -V | tail -1)
apt-get install -y -qq --no-install-recommends "$NCU_PKG" > /dev/null 2>&1
NCU=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)
echo "installed $NCU_PKG: $NCU"
export RUN_CACHE=$B/cache HF_HOME=$B/hf PYTHONPATH=$B/$PKG
export LD_LIBRARY_PATH=$B/venv/lib/python3.12/site-packages/nvidia/cublas/lib
for kv in "$@"; do export "$kv"; done
R=$PWD/root; mkdir -p "$R"; cp $B/$RUNNER/*.py "$R/"; cp $B/runner/units.json "$R/"; ln -sf $B/hf "$R/hf"
awk 'NF {print $1}' "$B/$LIST" | head -n "$N" > units.txt
$B/venv/bin/python -c "import json,sys; json.dump({'only_units': open(sys.argv[1]).read().split()}, open(sys.argv[2], 'w'))" \
  units.txt "$R/stop.json"
# Each kernel skips the launches of about <skip> decoding steps (launches a step from round19's profile; the
# encoder's kernels run once a layer a batch, about 3 a step).
i=0
for KS in cross_attention_kernel:32 Kernel2:134 kernel:182 tiled_split_gemm_kernel:29 residual_norm_kernel:102 \
          reorder_append_parts_kernel:32 copy_parts_kernel:32 split_heads_bias_kernel:64 warp_softmax_forward:177 \
          bias_add_vec_kernel:36 topk_stage_1:5 gemv2N_kernel:32 exact_attention_kernel:3 \
          exact_attention_layout:3 'regex:^(ampere|sm8|cutlass).*gemm':3; do
  K=${KS%:*} KSKIP=$((${KS##*:} * SKIP))
  i=$((i + 1))
  export RUN_OUT=$PWD/out$i                                  # a fresh folder: the runner skips units already written
  mkdir -p "$RUN_OUT"
  (cd "$R" && timeout 900 "$NCU" --target-processes all --kernel-name-base function -k "$K" \
    --launch-skip "$KSKIP" --launch-count "$COUNT" \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats \
    --section LaunchStats -f -o "$OLDPWD/report$i" $B/venv/bin/python transcribe_run.py back "${MODE:-stream8}" \
    > "$OLDPWD/ncu$i.out" 2>&1)
  echo "$K: exit $?, $(ls report$i.ncu-rep 2>/dev/null || echo no report)"
done
METRICS=gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct,launch__grid_size,launch__block_size,launch__registers_per_thread,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio
for f in report*.ncu-rep; do
  "$NCU" --import "$f" --page raw --csv --metrics "$METRICS" > "${f%.ncu-rep}.csv" 2>/dev/null
done
wc -l report*.csv
$B/venv/bin/python - report*.csv <<'EOF'
import csv, sys, collections
by, head = collections.defaultdict(list), None
for path in sys.argv[1:]:
    rows = list(csv.reader(open(path)))
    if len(rows) < 3:
        continue
    head = rows[0]
    k = head.index("Kernel Name")
    for r in rows[2:]:
        by[r[k]].append(r)
short = lambda m: m.replace("smsp__average_warps_issue_stalled_", "stall_").replace("_per_issue_active.ratio", "") \
    .replace(".avg.pct_of_peak_sustained_elapsed", "%").replace(".avg.pct_of_peak_sustained_active", "%")
cols = [c for c in head if c.startswith(("gpu__", "dram__", "lts__", "sm__", "launch__", "smsp__"))] if head else []
for name, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    vals = []
    for c in cols:
        i = head.index(c)
        try:
            xs = [float(r[i].replace(",", "")) for r in rs]
            vals.append(f"{short(c)}={sum(xs) / len(xs):.3g}")
        except ValueError:
            pass
    print(f"{name[:40]} x{len(rs)}: " + " ".join(vals))
EOF
