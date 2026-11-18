# mx-burner

GPU 算力基准与烤机工具。一套代码，两条后端：**NVIDIA CUDA** 与 **沐曦 MACA（曦云 C550）**。

最初是纯 CUDA 版本（CUDABurner）。现在多了：沐曦编译适配、运行时矩阵尺寸探测、内存与互连带宽测试、
逐项错误隔离。CUDA 路径的算子与测量逻辑没有改动。

C550 ×8 实测：FP16 dense **277.9 TFLOPS**、INT8 **387.8 TOPS**、八卡同压总功耗 **3576 W**。

---

## 1. 实测结果

单卡（autotune 自动选边长）：

| 精度 | 结果 | 选用边长 |
|---|---|---|
| FP64 | 4.8 TFLOPS | 2048 |
| FP32 | 36.3 TFLOPS | 8192 |
| TF32 | 129.1 TFLOPS | 8192 |
| FP16 | 277.9 TFLOPS | 16384 |
| FP16 (2:4) | 277.1 TFLOPS | 8192 |
| BF16 | 277.7 TFLOPS | 16384 |
| INT8 | 397.6 TOPS | 16384 |
| INT8 (2:4) | 不支持 | — |

八卡 FP16 并行（8 进程各占一卡，同时运行）：

| GPU | Dense | "Sparse"（见 4.3） |
|---|---|---|
| 0 | 270.47 | 449.02 |
| 1 | 262.23 | 448.99 |
| 2 | 259.46 | 448.79 |
| 3 | 265.65 | 448.92 |
| 4 | 260.17 | 449.11 |
| 5 | 262.15 | 448.96 |
| 6 | 262.15 | 448.93 |
| 7 | 261.55 | 448.95 |
| **均值** | **262.98** | **448.96** |

单卡 269.6 vs 八卡并发 263.0 的 2.5% 差异来自功耗耦合（8 卡同时贴 450 W 上限），不是预热不足。

内存与互连：HBM D2D **960.6 GB/s**，PCIe H2D/D2H 50.4/56.6 GB/s，P2P 55.3 GB/s。

完整数据（逐卡、精度矩阵、尺寸扫描、预热敏感性、复现命令）见 **[docs/C550_RESULTS.md](docs/C550_RESULTS.md)**。

## 2. 环境

沐曦路径：

- MACA 运行时，`/opt/maca`；编译用 `/opt/maca/tools/cu-bridge/bin/cucc`
- 头文件：`nvml.h`、`cuda_runtime.h`、`cublas_v2.h`、`cublasLt.h`、`cuda_fp16.h`、`cuda_bf16.h`、`cuda_fp8.h`
- 库：`libcudart` / `libmcblas` / `libmcblasLt`（`/opt/maca/lib`）、`libmxsml`（`/opt/mxdriver/lib`）
- 不需要安装 CUDA Toolkit，也不需要 NVIDIA 驱动

CUDA 路径沿用 `CMakeLists.txt`（`nvcc` + `libnvidia-ml`）。

实测机配置：

| 项 | 值 |
|---|---|
| GPU | 8 × MetaX C550（MXC550），单卡 65536 MiB，计算能力 8.0，104 SM |
| 卡间 | 8 卡两两 MetaXLink；GPU0-3 属 NUMA0，GPU4-7 属 NUMA1 |
| 驱动 | 3.4.3，MACA 3.7.1.5，`mx-smi 2.3.1`，BIOS 1.30.0.0 |
| CPU | 2 × Xeon Platinum 8468V，96 核 / 192 线程，800–3800 MHz |
| 内存 | 1.0 TiB，无 swap |
| OS | Ubuntu 22.04，gcc 11.4.0，kernel 5.14.0-3.1.8.kwai.x86_64 |

功耗温度：待机 93 W / 36 °C；满载 451 W（墙 450 W）/ 64 °C；八卡同压总 3576 W。

## 3. 构建与运行

```bash
./scripts/build.sh                                  # cucc 构建，默认用满核（JOBS=8 可覆盖）
./scripts/run.sh --mode benchmark                    # 全部精度，尺寸自动探测
./scripts/run.sh --mode benchmark --precision TF32 FP16
./scripts/run.sh --mode benchmark --device 3         # 指定卡
./scripts/run.sh --mode benchmark --size legacy    # 用 2048/4096/8192 这组固定边长
./scripts/run.sh --mode benchmark --size 8192        # 强制统一边长
./scripts/run.sh --mode memory --device 0            # HBM / PCIe / P2P 带宽
./scripts/run.sh --mode stress                       # 烤机
./scripts/run8.sh benchmark FP16                     # 8 卡并行，日志在 runs/
```

程序在测试结束后停在 `Press Ctrl+C to exit.`，脚本里用轮询 `Test finished` + `SIGINT` 收尾。

## 4. 测量口径

结果数字依赖测量方式，这里有四处与直觉不同，看数前请先读。

### 4.1 矩阵尺寸自动探测

矩阵边长直接决定吞吐。C550 上同一份 FP16 代码，边长 2048 只有 175 TFLOPS，16384 是 278 TFLOPS。
所以每项测试前会自动扫 `{2048, 4096, 8192, 16384}`（FP64 上限 4096），取实测最高者，
在 `Engine / Notes` 里标 `@8192` 这样的后缀。

探测与正式测试同构：每 8 次提交同步一次，跑满 1 秒。不能用固定迭代数探测——TF32 在 16384 上
短测（8 次）是 120 TFLOPS，跑满 10 秒只有 60.4，短测会选错边长。

`--size legacy` 回到固定边长（FP64 2048 / 其余 4096 / INT8 8192），便于和历史数据对比。

### 4.2 计时方式

每个精度项跑 10 秒，但**每 16 次提交同步一次**，队列深度有界。

原先是「异步提交满 10 秒，最后同步一次」。C550 上单次提交 4.7 μs、单次 GEMM 0.51 ms，
10 秒能提交 212 万次、GPU 要 1085 秒才算完 —— 一项实际耗时 18 分钟。现在 2 项 38 秒。

### 4.3 稀疏项不是真稀疏

`gemm_sparse_tensorcore.cu` 只设了转置属性，没有 2:4 压缩矩阵，也没有 `CUBLASLT_MATMUL_DESC_SPARSE`
（沐曦 cu-bridge / mcblasLt 都没提供该属性）。实际跑的是 dense 的转置 GEMM。

所以稀疏条目的数值按 dense 记账，不再乘 2。八卡表里的 448.96 是旧口径的虚报，真值约 224。
换到支持该属性的后端时，还需要额外生成压缩矩阵才叫真稀疏。

### 4.4 精度可用性

`cublasGemmEx` 在 C550 上接受：FP64 / FP32 / TF32 / BF16 / FP16 / INT8。
拒绝（`status 15`）：`COMPUTE_32F_FAST_16F/16BF`、FP8 E4M3/E5M2、FP6 E2M3/E3M2、FP4 E2M1、INT4。

注意 `library_types.h` 里这些枚举都定义了（`MACA_R_8F_E4M3`、`_6F_E2M3`、`_4F_E2M1`、`_4I`…），
但实现不接受——头文件有枚举不等于能用。判断能力必须实际调用。

## 5. 平台差异

| 项 | 说明 |
|---|---|
| FP4 | 沐曦没提供 `cuda_fp4.h`，且 C550 `cc=8.0` 不满足 FP4 分支条件（需 9.0/8.9）。`compat/gemm_fp4_unimplemented.cpp` 提供抛异常的桩，保证链接通过且不静默出错 |
| FP8 | 编译通过，但 `cc=8.0` 使 `operator_factory` 走 FP32 模拟分支，运行时不达 |
| GfxClock | `nvmlDeviceGetClockInfo(NVML_CLOCK_GRAPHICS)` 返回不支持，代码失败即置 0 |
| FP64 | 4.8 TFLOPS，`GemmGeneric<double>` 首次调用慢（库侧 kernel 选择），第一项要给足时间 |
| 单项失败 | 某项抛异常（如 INT8 稀疏）只标记为 unsupported，不影响后续项 |

## 6. 目录

| 路径 | 内容 |
|---|---|
| `main.cu`、`core/`、`operators/`、`strategies/`、`ui/`、`utils/` | 算子、策略、TUI |
| `monitoring/memory_bench.*` | HBM / PCIe / P2P 带宽测试 |
| `compat/gemm_fp4_unimplemented.cpp` | FP4 桩（沐曦无 `cuda_fp4.h`） |
| `Makefile` | 沐曦构建（cucc） |
| `CMakeLists.txt` | CUDA 构建（nvcc） |
| `scripts/` | build / run / run8 |
| `tools/` | 精度能力探测、尺寸扫描、内存探测、预热敏感性 |
| `docs/C550_RESULTS.md` | C550 完整实测数据 |
| `IMPLEMENTATION.md` | 沐曦适配的实现依据（行号级证据） |
| `docs/README.cudaburner.md` | CUDA 版本原始说明 |
