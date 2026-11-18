# MetaX C550 ×8 实测结果

采集时间：2026-11-18
采集方式：`mx-burner`（CUDABurner 移植版）+ 独立探测程序，全部在目标机上原样跑出
机器：KML dtmachine `100001062`

---

## 1. 系统配置

| 项 | 值 |
|---|---|
| GPU | 8 × MetaX C550（MXC550 / 曦云 C550） |
| 单卡显存 | 65536 MiB（`cudaGetDeviceProperties` 报 total_memory 63.6 GiB） |
| 计算能力 | **8.0**（major=8, minor=0），104 个多处理器 |
| 卡间互连 | `mx-smi topo -m` 显示 8 卡两两均为 **MX = MetaXLink**；GPU0-3 属 NUMA0，GPU4-7 属 NUMA1 |
| 驱动 | Kernel Mode Driver **3.4.3**，MACA **3.7.1.5**，`mx-smi 2.3.1`，BIOS 1.30.0.0 |
| CPU | 2 × Intel Xeon Platinum 8468V，48C/96T ×2 = **96 核 / 192 线程**，800–3800 MHz，2 NUMA |
| 内存 | **1.0 TiB**，无 swap |
| OS / 工具链 | Ubuntu 22.04，gcc 11.4.0，kernel 5.14.0-3.1.8.kwai.x86_64 |
| 编译 | `cucc`（`mxcc version 1.0.0`），`make -j96` |

待机 vs 满载（NVML/libmxsml 实测）：

| 指标 | 待机 | 满载 |
|---|---|---|
| 温度 | 36 °C | 64 °C |
| 功耗 | 93 W | 451 W（功耗上限 450 W） |
| GPU 利用率 | 0 % | 98–99 % |
| 显存频率 | 1800 MHz | 1800 MHz |
| 图形频率 | 读不到（`NVML_CLOCK_GRAPHICS` 返回不支持） | 同左 |

---

## 2. 八卡 FP16 测试（CUDABurner 原版 `--mode benchmark --precision FP16`）

8 个独立进程，`MACA_VISIBLE_DEVICES=0..7` 各占一卡，同时运行。
每项测试 = 上游逻辑（提交 10 s + synchronize），实测每项约 18 分钟。

| GPU | FP16 Dense (TFLOPS) | "Sparse" (TFLOPS) | 备注 |
|---|---|---|---|
| 0 | **270.47** | 449.02 | |
| 1 | 262.23 | 448.99 | |
| 2 | 259.46 | 448.79 | |
| 3 | 265.65 | 448.92 | |
| 4 | 260.17 | 449.11 | |
| 5 | 262.15 | 448.96 | |
| 6 | 262.15 | 448.93 | 上游笔误打印 "TFLOPS"→"TFLOPS" 原文为 `TFLOPS` |
| 7 | 261.55 | 448.95 | |
| 均值 | **262.98** | **448.96** | |

- Dense 卡间离散度约 ±2%（`262.98 ± 5.4`），首卡略高（8 卡同时跑时 GPU0 拿到最多前端带宽）。
- Sparse 8 卡误差仅 **±0.3%**，异常整齐 —— 见第 5.1 节，这是因为它并未真正启用稀疏。
- 修正记账后同一项在单卡上为 **223.56 TFLOPS**（dense 269.86），稀疏/dense 比值 <1，符合"转置布局 + 同一 dense kernel"。

同时刻 GPU 状态（8 卡满载）：

```
g0:451/450W@58C g1:451/450W@58C g2:442/450W@57C g3:450/450W@56C
g4:451/450W@58C g5:451/450W@56C g6:452/450W@58C g7:451/450W@55C
总功耗 3576 W
```

---

## 3. 单卡微基准（排除 TUI/队列干扰）

4096³ 矩阵乘，预热后取 20 次平均：

| 项 | 值 |
|---|---|
| FP16 `cublasGemmEx` 单次 | **0.51 ms** |
| 对应吞吐 | **270.3 TFLOPS** |
| CPU 侧提交开销 | 4.7 μs/次 |

270.3 TFLOPS 与同机 PyTorch 直测的 274 TFLOP/s 一致 → 走真 Tensor Core，非模拟。

---

## 4. 全精度能力矩阵（`cublasGemmEx`，2048³）

| 精度 | A/B 类型 | C 类型 | compute type | 结果 |
|---|---|---|---|---|
| FP64 | `CUDA_R_64F` | 64F | `COMPUTE_64F` | ✅ **4.6 TFLOPS** |
| FP32 | `CUDA_R_32F` | 32F | `COMPUTE_32F` | ✅ **34.6 TFLOPS** |
| TF32 | `CUDA_R_32F` | 32F | `COMPUTE_32F_FAST_TF32` | ✅ **83.1 TFLOPS** |
| BF16 | `CUDA_R_16BF` | 16BF | `COMPUTE_32F` | ✅ **174.1 TFLOPS** |
| BF16 (C=f32) | `CUDA_R_16BF` | 32F | `COMPUTE_32F` | ✅ 168.8 TFLOPS |
| FP16 | `CUDA_R_16F` | 16F | `COMPUTE_32F` | ✅ **176.3 TFLOPS** |
| FP16 (C=f32) | `CUDA_R_16F` | 32F | `COMPUTE_32F` | ✅ 168.9 TFLOPS |
| INT8 | `CUDA_R_8I` | 32I | `COMPUTE_32I` | ✅ **202.8 TOPS** |
| FP16 | `CUDA_R_16F` | 32F | `COMPUTE_32F_FAST_16F` | ❌ status 15 |
| BF16 | `CUDA_R_16BF` | 32F | `COMPUTE_32F_FAST_16BF` | ❌ status 15 |
| FP8 E4M3 | `CUDA_R_8F_E4M3` | 16F / 32F | `COMPUTE_32F` | ❌ status 15 |
| FP8 E5M2 | `CUDA_R_8F_E5M2` | 16F | `COMPUTE_32F` | ❌ status 15 |
| FP6 E2M3 | `CUDA_R_6F_E2M3` | 16F | `COMPUTE_32F` | ❌ status 15 |
| FP6 E3M2 | `CUDA_R_6F_E3M2` | 16F | `COMPUTE_32F` | ❌ status 15 |
| FP4 E2M1 | `CUDA_R_4F_E2M1` | 16F | `COMPUTE_32F` | ❌ status 15 |
| INT4 | `CUDA_R_4I` | 32I | `COMPUTE_32I` | ❌ status 15 |

`status 15` = `CUBLAS_STATUS_NOT_SUPPORTED`，mcblas 打印：

```
[MCBLAS][E]Sorry, the combination of the parameters Atype, Btype and Ctype is not supported
```

> **注意**：`library_types.h` 里 `MACA_R_8F_E4M3` / `_8F_E5M2` / `_6F_E2M3` / `_6F_E3M2` /
> `_4F_E2M1` / `_4I` / `_8F_UE8M0` 这些枚举**都已定义**，但 mcblas 的 GEMM 不接受它们。
> 即：头文件对齐了 CUDA 的类型表，实现尚未覆盖 —— 不能凭头文件里的枚举判断能力。
> 未验证是否可通过 `cublasLtMatmul` + 显式 scale 走通（MXFP8 一般需要这条路径）。

2048³ 的绝对值低于 4096³，是因为小矩阵利用率低（FP16 2048³ 176 TFLOPS vs 4096³ 270 TFLOPS）。
该表用于**判断"能不能跑"**，不是峰值。

---

## 5. 发现的两个工具问题

### 5.1 "Sparse" 项并未真正启用稀疏

`operators/gemm_sparse_tensorcore.cu` 只设置了转置属性，**没有设置稀疏属性**：

```cpp
cublasLtMatmulDescCreate(&matmul_desc_, compute_type, CUDA_R_32F);
cublasOperation_t op_trans_a = CUBLAS_OP_T;
cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &op_trans_a, sizeof(op_trans_a));
// 缺 CUBLASLT_MATMUL_DESC_SPARSE
```

而吞吐记账却按稀疏算了双倍：

```cpp
gflops_or_gops_ = (2.0 * m_ * n_ * k_ * 2.0) / 1e9;   // 末尾 *2 是稀疏加成
```

因此 `448.96 TFLOPS` 是 **dense 值 × 2 的虚报**。真实吞吐约 224 TFLOPS
（低于 dense 的 263，符合"转置布局 + 同一 kernel"的预期）。
8 卡 ±0.3% 的高度一致也印证了它们跑的是同一条 dense 路径。

要真正测稀疏（2:4 结构化），需要：压缩后的 A 矩阵（每 4 个元素留 2）+ `CUBLASLT_MATMUL_DESC_SPARSE`。

### 5.2 每项测试耗时 ≈ 18 分钟，不是 15 秒

`strategies/benchmark_strategy.cpp` 的计时逻辑是「异步提交 10 秒 → synchronize」：

```cpp
while (elapsed_seconds < 10.0) { op->execute(0); iterations++; ... }
cudaDeviceSynchronize();
```

| 量 | 值 |
|---|---|
| 单次 GEMM（4096³ FP16） | 0.51 ms |
| CPU 提交开销 | 4.7 μs |
| 10 秒内可提交 | ≈ 212 万次 |
| GPU 算完需要 | ≈ 1085 秒（18 分钟） |

即 10 秒窗口在这个算力档位上等价于 18 分钟队列。8 项全跑约 2.5 小时。

---

## 6. 内存子系统（额外探测）

| 项 | 实测 | 说明 |
|---|---|---|
| HBM D2D copy | **960.6 GB/s** | 256 MiB，0.56 ms（按读+写计） |
| PCIe H2D（pinned） | 50.4 GB/s | |
| PCIe D2H（pinned） | 56.6 GB/s | |
| P2P 0→1（同 NUMA） | **55.3 GB/s** | `cudaMemcpyPeer` |
| P2P 0→7（跨 NUMA） | **55.2 GB/s** | 跨 NUMA 无衰减 |
| P2P 可访问性 | 0→1、1→0、0→7 均为 1 | `cudaDeviceCanAccessPeer` |

> **未确认**：P2P 实测 55 GB/s 与厂商标称「八卡全互连 896 GB/s」差距很大。
> `cudaMemcpyPeer` 很可能走的是 PCIe 而非 MetaXLink；MetaXLink 的真实带宽需通过
> MXCCL（`libmccl`）的 all-to-all 测试来验证，本次未做。

---

## 6.5 修正后的单卡全精度实测（mx-burner）

`./CUDABurner --mode benchmark --device 0 --precision FP64 FP32 TF32 FP16 BF16 INT8`

| 精度 | 模式 | 结果 | Engine |
|---|---|---|---|
| FP64 | Dense | **4.57 TFLOPS** | CUDA Core (Native) |
| FP32 | Dense | **35.63 TFLOPS** | CUDA Core (TC Disabled) |
| TF32 | Dense | **106.45 TFLOPS** | Tensor Core (Default FP32) |
| FP16 | Dense | **269.63 TFLOPS** | Tensor Core |
| FP16 | "Sparse" | **223.68 TFLOPS** | 修正后按 dense 记账（原 449 为虚报） |
| BF16 | Dense | **269.12 TFLOPS** | Tensor Core |
| INT8 | Dense | **376.11 TOPS** | Tensor Core |
| INT8 | Sparse | N/A | `unsupported on this backend: cuBLASLt Error - 15` |

要点：

- 进程输出 `CUDABurner finished cleanly` —— 逐项 try/catch 生效，单项失败不再终止整轮
- FP16 Dense 269.63 与前一轮 262.98（8 卡并发）略高，因为单卡独占
- INT8 376.11 TOPS 明显高于 2048³ 的 202.8，因为上游 INT8 用 8192³，利用率更高
- FP16 "Sparse" 223.68 < Dense 269.63，符合"转置布局 + 同一 dense kernel"的预期

## 6.8 预热是否足够（敏感性实验）

上游每项测试前预热 **10 次**（`for (int i = 0; i < 10; ++i) op->execute(0)`），
在 C550 上折算仅 **5.1 ms**。担心点：GPU 的 DVFS 升频、mcblas 的首次 kernel 选择是否会被打断。

方法：每个数据点用**独立进程**（真实冷启动），预热 N 次后测量 1 秒稳态吞吐
（每 16 次提交 sync 一次，排除队列积压）。矩阵 4096³ FP16。

| 预热次数 | 折算耗时 | 实测吞吐 |
|---|---|---|
| 0 | 0 ms | 265.3 TFLOPS |
| 1 | 0.51 ms | 269.6 TFLOPS |
| 3 | 1.5 ms | 269.7 TFLOPS |
| **10（上游取值）** | **5.1 ms** | **269.6 TFLOPS** |
| 50 | 25.5 ms | 269.6 TFLOPS |
| 200 | 102 ms | 269.6 TFLOPS |
| 1000 | 510 ms | 269.7 TFLOPS |

**结论：1 次预热即达稳态**（0 次仅低 1.6%），预热到 1000 次没有任何额外收益。
C550 在这条路径上没有可观测的升频延迟，mcblas 也没有 autotune 式的首次开销。

测量时长也不敏感（预热固定 10 次）：

| 测量窗口 | 吞吐 |
|---|---|
| 1 s | 269.7 TFLOPS |
| 3 s | 269.6 TFLOPS |
| 10 s（上游取值） | 269.7 TFLOPS |

**所以 10 秒窗口内不存在"边测边爬频"导致的低估。**

附带结论：单卡独占 269.6 vs 8 卡并发 263.0（差 2.5%），差异来自 8 卡同时满载的功耗/散热耦合
（实测总功耗 3576 W，各卡贴 450 W 上限），**不是预热不足**。

## 6.9 矩阵尺寸对吞吐的影响（上游尺寸并非最优）

上游硬编码：FP64 用 2048³，FP32/TF32/FP16/BF16 用 4096³，INT8 用 8192³。
实测（`tools/probe_size.cu`，每点预热 3 次测 8 次，4096³ 以上每点 >100 ms 已无预热问题）：

| dim | FP64 | FP32 | TF32 | BF16 | FP16 | INT8 |
|---|---|---|---|---|---|---|
| 1024 | 3.7 | 26.2 | 45.4 | 98.7 | 100.8 | 96.8 |
| 2048 | 4.6 | 34.5 | 83.0 | 164.5 | 175.1 | 200.0 |
| **4096（上游）** | **4.7** | **35.6** | **106.3** | **267.8** | **268.4** | 368.2 |
| 8192 | 4.8 | 36.3 | **129.4** | 274.2 | 274.9 | 378.9 |
| 16384 | – | 35.8 | **120.5 ↓** | **277.7** | **278.0** | **397.6** |

要点：

- **TF32 收益最大**：4096³ → 8192³ 从 106.3 涨到 129.4（**+22%**）
- **TF32 在 16384³ 反而下降**到 120.5 —— 大边长超出某个 cache/tile 优化区间
- FP64 在 2048³ 即饱和（4.6→4.8），上游选择正确
- FP32 在 8192³ 饱和（36.3），4096³ 已接近
- INT8 一直涨到 16384³（397.6，比上游 8192³ 的 378.9 高 5%）

## 6.10 运行时自动探测（autotune）

因为最优尺寸随平台/库版本变化，硬编码标定值不可持续。`benchmark_strategy` 在每项正式开始前
**自动扫描候选边长 {2048, 4096, 8192, 16384}**（FP64 上限 4096），取实测吞吐最高者，结果缓存。

探测本身与正式测试同构：每 8 次提交 sync 一次，跑满 **1 秒**。

> 关键教训：最初用「固定 8 次迭代」探测，TF32 误选 16384（短测 ~120，长测只有 **60.41**）。
> 短测不等于长测稳态，探测必须与正式测试同构。

实测（`--mode benchmark --device 0 --precision TF32 FP16 INT8`，未指定 `--size`）：

| 项 | autotune 选择 | 实测 | 上游 4096³ 对比 |
|---|---|---|---|
| TF32 Dense | **8192** | **129.05 TFLOPS** | 106.3（+21%） |
| FP16 Dense | 16384 | 277.87 TFLOPS | 268.4（+3.5%） |
| FP16 Sparse | 8192 | 277.14 TFLOPS | — |
| INT8 Dense | 16384 | 387.84 TOPS | — |

控制方式：

```bash
./CUDABurner --mode benchmark --precision TF32          # 默认 autotune
./CUDABurner --mode benchmark --size upstream            # 复现上游尺寸（可比性）
./CUDABurner --mode benchmark --size 8192                # 强制统一边长
```

## 7. 复现命令

```bash
# 构建
./scripts/build.sh

# 单卡：全部精度
./scripts/run.sh --mode benchmark

# 单卡：指定精度
./scripts/run.sh --mode benchmark --precision FP16 BF16 TF32 INT8

# 八卡：每卡一进程，日志在 runs/
./scripts/run8.sh benchmark FP16

# 精度能力探测（本仓库 tools/）
cucc tools/probe_precision.cu -o probe_precision \
     -I/opt/maca/tools/cu-bridge/include \
     -L/opt/maca/lib -L/opt/mxdriver/lib -lcudart -lmcblas
./probe_precision

# 内存/P2P 探测
cucc tools/probe_memory.cu -o probe_memory \
     -I/opt/maca/tools/cu-bridge/include \
     -L/opt/maca/lib -L/opt/mxdriver/lib -lcudart
./probe_memory
```
