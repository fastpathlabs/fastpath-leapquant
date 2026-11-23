#!/usr/bin/env bash
# g3_confirm.sh: gate G3: per token at batch 512 (one layer, decode plus 1/16 of a
# fold) at most 0.8 x the bf16 + window kernel's (SGLang's replay window kernel, bf16 state), paired, 10 interleaved
# repetitions. Each repetition is one bench_triton.py run that times LeapQuant and SGLang's kernels in the same
# process; the arms (configurations) are interleaved inside each repetition.
#   RESULTS=results/g3 REPS=10 ARMS="adopted sweeps1 v8" runbook/g3_confirm.sh     (from the repository root)
# Arms: adopted = CUDA fold (batched Gram-Schmidt) + Triton decode, 2 sweeps; sweeps1 = the same with one sweep
# (B52, a policy option); v8 / v9 = CUDA fold + CUDA decode variant 8 / 9; v9sweeps1 = v9 with one sweep.
# Output: RESULTS/<arm>_<rep>.txt, RESULTS/G3.txt.
set -u
cd "$(dirname "$0")/.."
source "${VENV_SGLANG:-$HOME/venv-sglang}/bin/activate"
mkdir -p "$RESULTS"
runbook/manifest.sh "$RESULTS/manifest.txt"
B="python3 kernel/bench_triton.py --batch-sizes 512 --windows 2 --no-exact-times --fold-backend cuda --with-sglang"
for rep in $(seq 1 "${REPS:-10}"); do
    for arm in ${ARMS:-adopted}; do
        case "$arm" in
            adopted) timeout 900 $B --decode-backend triton > "$RESULTS/${arm}_$rep.txt" 2>&1 ;;
            sweeps1) timeout 900 $B --decode-backend triton --sweeps 1 > "$RESULTS/${arm}_$rep.txt" 2>&1 ;;
            v8) LEAPQUANT_CUDA_DECODE_VARIANT=8 timeout 900 $B --decode-backend cuda > "$RESULTS/${arm}_$rep.txt" 2>&1 ;;
            v9) LEAPQUANT_CUDA_DECODE_VARIANT=9 timeout 900 $B --decode-backend cuda > "$RESULTS/${arm}_$rep.txt" 2>&1 ;;
            v9sweeps1) LEAPQUANT_CUDA_DECODE_VARIANT=9 timeout 900 $B --decode-backend cuda --sweeps 1 > "$RESULTS/${arm}_$rep.txt" 2>&1 ;;
        esac
    done
done
python3 - "$RESULTS" ${ARMS:-adopted} > "$RESULTS/G3.txt" <<'EOF'
import glob, math, os, re, statistics, sys
folder, arms = sys.argv[1], sys.argv[2:]
print("G3: LeapQuant per token against bf16 + window per token, batch 512, paired within each run (gate: ratio >= 1.25)")
for arm in arms:
    ratios, lq, bf = [], [], []
    for path in sorted(glob.glob(os.path.join(folder, f"{arm}_*.txt"))):
        text = open(path).read()
        m = re.search(r"->\s+([\d.]+) ms per token", text)
        w = re.search(r"sglang window kernel, bf16 state: ([\d.]+) ms per token", text)
        if m and w:
            lq.append(float(m.group(1))); bf.append(float(w.group(1))); ratios.append(bf[-1] / lq[-1])
    if len(ratios) < 2:
        print(f"{arm:8} too few runs ({len(ratios)})"); continue
    mean, sd = statistics.mean(ratios), statistics.stdev(ratios)
    t975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
            12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 20: 2.093, 30: 2.045}
    n = len(ratios)
    half = t975.get(n, 2.0 if n > 30 else t975[min(k for k in t975 if k >= n)]) * sd / math.sqrt(n)
    verdict = "MET" if mean - half >= 1.25 else ("NOT MET" if mean + half < 1.25 else "INCONCLUSIVE")
    print(f"{arm:8} n={len(ratios)}  LeapQuant {statistics.mean(lq):.4f} ms  bf16 window {statistics.mean(bf):.4f} ms  "
          f"ratio {mean:.3f} (95% {mean - half:.3f} .. {mean + half:.3f})  {verdict}")
EOF
cat "$RESULTS/G3.txt"
