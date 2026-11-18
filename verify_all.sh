#!/bin/bash
cd /tmp/mxb2
echo "########## 方向3+4：memory 模式 + --device 1 ##########"
./CUDABurner --mode memory --device 1 2>&1 | tail -22

echo
echo "########## 方向2+1：benchmark on device 2，FP16（含 sparse 记账） ##########"
START=$(date +%s)
setsid ./CUDABurner --mode benchmark --device 2 --precision FP16 > bench_fp16.log 2>&1 &
PID=$!
for i in $(seq 1 40); do
  sleep 3
  if sed "s/\x1b\[[0-9;?]*[a-zA-Z]//g" bench_fp16.log | grep -aq "Test finished"; then break; fi
done
kill -INT $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null
END=$(date +%s)
echo "wall=$((END-START))s"
sed "s/\x1b\[[0-9;?]*[a-zA-Z]//g" bench_fp16.log | grep -aE "^(FP|INT)[0-9]+ +(Dense|Sparse)|mx-burner|GPU Name|Test finished" | sort -u | head -12
echo VERIFY_DONE
