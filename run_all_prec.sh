#!/bin/bash
cd /tmp/mxb2
echo "start $(date +%H:%M:%S)"
START=$(date +%s)
setsid ./CUDABurner --mode benchmark --device 0 --precision FP64 FP32 TF32 FP16 BF16 INT8 > allprec.log 2>&1 &
PID=$!
for i in $(seq 1 100); do
  sleep 3
  if sed "s/\x1b\[[0-9;?]*[a-zA-Z]//g" allprec.log | grep -aq "Test finished"; then break; fi
done
kill -INT $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null
END=$(date +%s); echo "wall=$((END-START))s"
sed "s/\x1b\[[0-9;?]*[a-zA-Z]//g" allprec.log | grep -aE "^(FP|INT)[0-9]+ +(Dense|Sparse)" | sort -u
echo ALLPREC_DONE
