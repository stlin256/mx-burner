#!/bin/bash
# 单卡运行；透传上游参数
#   ./scripts/run.sh                        # 交互菜单
#   ./scripts/run.sh --mode benchmark       # 全部精度
#   ./scripts/run.sh --mode benchmark --precision FP16 BF16
#   ./scripts/run.sh --mode stress
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x ./CUDABurner ] || { echo "先运行 ./scripts/build.sh" >&2; exit 1; }
exec ./CUDABurner "$@"
