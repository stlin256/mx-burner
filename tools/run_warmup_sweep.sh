#!/bin/bash
cd "$(dirname "$0")"
echo "== 每次都是独立进程（冷启动），测量 1 秒"
for n in 0 1 3 10 50 200 1000; do
  ./probe_warmup $n 1.0
done
echo "== 固定 warmup=10，测不同时长（看 10 秒窗口内是否仍在爬升）"
for s in 1 3 10; do
  ./probe_warmup 10 $s
done
echo SWEEP_DONE
