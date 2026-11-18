#!/bin/bash
cd /tmp/mxb5
run_one() {
  local tag="$1"; shift
  setsid ./CUDABurner "$@" > sz_$tag.log 2>&1 &
  local pid=$!
  for i in $(seq 1 50); do
    sleep 3
    sed "s/\x1b\[[0-9;?]*[a-zA-Z]//g" sz_$tag.log | grep -aq "Test finished" && break
  done
  kill -INT $pid 2>/dev/null; sleep 2; kill -9 $pid 2>/dev/null
  echo "--- $tag"
  sed "s/\x1b\[[0-9;?]*[a-zA-Z]//g" sz_$tag.log | grep -aE "TF32 +Dense|FP16 +(Dense|Sparse)|INT8 +Dense" | sed "s/^ *//" | sort -u
}
echo "===== A) autotune 默认（未给 --size）"
run_one autotune --mode benchmark --device 0 --precision TF32 FP16 INT8
echo "===== B) 强制上游尺寸"
run_one upstream --mode benchmark --device 0 --precision TF32 FP16 INT8 --size upstream
echo SIZE_VERIFY_DONE
