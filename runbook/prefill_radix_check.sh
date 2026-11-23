#!/usr/bin/env bash
# prefill_radix_check.sh: why LeapQuant's prefill is slower with the radix cache on. results/ab-trackflush: per
# 8,192-token prefill batch LeapQuant ~17k prompt tokens/s against bf16 window's ~21k with the radix cache on; with it
# off both ~23-25k (rounds 12, 13), and time to first token is equal. Prefill-only bursts (eval/prefill_burst.py: 256
# requests x 128 random tokens, 4 new tokens, 5 bursts after a warm-up, cache flushed before each) on four servers
# with 1,100 state slots each: bf16 window and LeapQuant (compressed, CUDA fold and decode), radix cache on and off.
# The torch profiler of each server records the steps of burst 2 (profile_<label>/).
#   RESULTS=results/prefill-radix runbook/prefill_radix_check.sh          (from the repository root)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
GRAPHS="--cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-512}"
RADIX="--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule"
WINDOW="--enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton"
LIMITS="--max-running-requests 512 --max-mamba-cache-size ${SLOTS:-1100}"

step() {           # step NAME "description" command...
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
burst() {          # burst LABEL server-flags...
    local label="$1"; shift
    mkdir -p "$RESULTS/profile_$label"
    start_server "$RESULTS/server_$label.log" "$@" || return 1
    python3 eval/prefill_burst.py --port "$PORT" --out "$RESULTS/burst_$label.json" \
        --profile-dir "$RESULTS/profile_$label"
    local status=$?
    sleep 10                                       # let the profiler finish writing its trace
    ls -la "$RESULTS/profile_$label"
    return $status
}
lq_env() { export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}"; }
bf16_on()  { burst bf16_on  --mamba-ssm-dtype bfloat16 $RADIX $WINDOW $GRAPHS $LIMITS; }
bf16_off() { burst bf16_off --mamba-ssm-dtype bfloat16 $RADIX $WINDOW $GRAPHS $LIMITS --disable-radix-cache; }
lq_on()    { lq_env; burst lq_on  --mamba-ssm-dtype float32 $RADIX $WINDOW --enable-linear-leapquant $GRAPHS $LIMITS; }
lq_off()   { lq_env; burst lq_off --mamba-ssm-dtype float32 $RADIX $WINDOW --enable-linear-leapquant $GRAPHS $LIMITS --disable-radix-cache; }
step B1 "prefill bursts: bf16 window, radix cache on" bf16_on
step L1 "prefill bursts: LeapQuant, radix cache on" lq_on
step B0 "prefill bursts: bf16 window, radix cache off" bf16_off
step L0 "prefill bursts: LeapQuant, radix cache off" lq_off
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
