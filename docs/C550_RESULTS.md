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

| GPU | FP16 Dense (TFLOPS) | FP16 Sparse (TFLOPS) | 备注 |
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
- Sparse 8 卡误差仅 **±0.3%**，异常整齐 —— 见第 5 节，这是因为它并未真正启用稀疏。

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
