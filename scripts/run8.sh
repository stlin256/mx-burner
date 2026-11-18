#!/bin/bash
# 8 卡并行：每卡一个独立进程，通过 MACA_VISIBLE_DEVICES 隔离。
# 上游 main.cu 的 device_id 固定为 0，因此用进程级隔离而不是改源码。
#   ./scripts/run8.sh benchmark FP16     # 8 卡都跑 benchmark/FP16
#   ./scripts/run8.sh stress             # 8 卡都烤机
set -uo pipefail
cd "$(dirname "$0")/.."
[ -x ./CUDABurner ] || { echo "先运行 ./scripts/build.sh" >&2; exit 1; }

MODE=${1:-benchmark}; shift || true
PREC=("$@")
mkdir -p runs
PIDS=()
for i in $(seq 0 7); do
    if [ "$MODE" = benchmark ] && [ ${#PREC[@]} -gt 0 ]; then
        MACA_VISIBLE_DEVICES=$i setsid ./CUDABurner --mode benchmark --precision "${PREC[@]}" \
            > runs/gpu$i.log 2>&1 &
    else
        MACA_VISIBLE_DEVICES=$i setsid ./CUDABurner --mode "$MODE" \
            > runs/gpu$i.log 2>&1 &
    fi
    PIDS+=($!)
    echo "gpu$i pid=$!"
done

# 上游跑完后会停在 "Press Ctrl+C to exit." 不退出，这里轮询该标志再收尾
echo "等待完成（标志：Test finished）。日志：runs/gpu*.log"
for _ in $(seq 1 360); do
    sleep 10
    n=0
    for i in $(seq 0 7); do
        sed "s/\x1b\[[0-9;?]*[a-zA-Z]//g" runs/gpu$i.log 2>/dev/null | grep -aq "Test finished" && n=$((n+1))
    done
    [ "$n" -ge 8 ] && { echo "全部完成"; break; }
done
for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null; done
sleep 3
for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done
echo "结果见 runs/gpu*.log（TUI 每帧重绘，取最后一帧的表格）"
