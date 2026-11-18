#!/bin/bash
# 一键构建（沐曦 MACA / cu-bridge）
set -euo pipefail
cd "$(dirname "$0")/.."

CB=${CBB:-/opt/maca/tools/cu-bridge}
if [ ! -x "$CB/bin/cucc" ]; then
    echo "找不到 cucc：$CB/bin/cucc" >&2
    echo "请确认 MACA 已安装，或用 CBB=/path/to/cu-bridge 指定" >&2
    exit 1
fi

MAKE=${MAKE:-make}
# 默认用满 CPU；用 -j 覆盖：JOBS=8 ./scripts/build.sh
JOBS=${JOBS:-$(nproc)}
echo "cucc  : $CB/bin/cucc"
echo "jobs  : $JOBS"
exec "$MAKE" -j"$JOBS" CBB="$CB" "$@"
