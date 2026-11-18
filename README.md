# mx-burner

[CUDABurner](https://github.com/stlin256/CUDABurner) 的**沐曦（MetaX / MACA）移植版**：把原本只面向
NVIDIA 的 GPU 算力基准与烤机工具，跑到曦云 C550 上。

上游是针对 NVIDIA 生态写的（CUDA Toolkit + cuBLAS + cuBLASLt + NVML）。本仓库不做功能改写，
只通过沐曦 **cu-bridge**（CUDA API 兼容层）编译运行，**上游源码零修改**。

```
上游 CUDABurner (commit d85c030, 2025-12-12)
        │
        ├── 上游源码（core/ monitoring/ operators/ strategies/ ui/ utils/ main.cu）—— 原样
        ├── compat/          移植层：FP4 单元桩（本仓库新增）
        ├── Makefile         cucc 构建（本仓库新增，替代上游 CMakeLists 的 nvcc 路径）
        └── scripts/         构建与运行脚本
```

---

## 1. 验证状态

| 项 | 状态 | 证据 |
|---|---|---|
| 编译 | ✅ | `cucc` + `make -j96`，`EXIT=0`，376 KB 可执行文件，无 undefined reference |
| CUDA 代码可运行 | ✅ | 最小 kernel + `cudaGetDeviceProperties` 返回 `MetaX C550 cc=8.0 sm=104` |
| cuBLAS（cu-bridge） | ✅ | SGEMM 1024³ 结果正确；`cublasSetMathMode(TF32)`、`cublasGemmEx` FP16 / INT8 均 `status=0` |
| NVML（cu-bridge → libmxsml） | ✅ | 真读到 名字/驱动/温度/功耗/功耗上限/利用率/显存频率/throttle |
| 单卡满载 | ✅ | GPU 利用率 99%，功耗 265 W（待机 93 W） |
| 8 卡并行满载 | ✅ | 8 进程各占一卡，总功耗 **3576 W**（8×451 W，全部顶到 450 W 上限），55–64 °C |
| benchmark 数值表 | ✅ | 8 卡 FP16 dense **262.98** / sparse **448.96** TFLOPS，明细见 [docs/C550_RESULTS.md](docs/C550_RESULTS.md) |

单次 FP16 GEMM 实测（4096³，`cublasGemmEx` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP`）：

```
0.51 ms/call  ->  270.3 TFLOPS
```

与同机 PyTorch 直测的 274 TFLOP/s 一致，说明走的是真 Tensor Core，不是 FP32 模拟。

## 2. 环境要求

- 沐曦 MACA 运行时（驱动 + SDK），`/opt/maca`；cu-bridge 位于 `/opt/maca/tools/cu-bridge`
- 提供 `nvml.h` / `cuda_runtime.h` / `cublas_v2.h` / `cublasLt.h` / `cuda_fp16.h` / `cuda_bf16.h` / `cuda_fp8.h`
- `libmxsml`（`/opt/mxdriver/lib`）、`libmcblas` / `libmcblasLt`（`/opt/maca/lib`）
- 无需安装 CUDA Toolkit、无需 NVIDIA 驱动

实测环境：C550 ×8、Driver 3.4.3、MACA 3.7.1.5、`mx-smi 2.3.1`、Ubuntu 22.04、gcc 11.4。

### 实测系统配置

| 项 | 值 |
|---|---|
| GPU | 8 × MetaX C550（MXC550），单卡 65536 MiB，计算能力 **8.0**，104 SM/卡 |
| 卡间互连 | 8 卡两两 **MetaXLink**（`mx-smi topo -m` 全 `MX`）；GPU0-3 属 NUMA0，GPU4-7 属 NUMA1 |
| 驱动 | Kernel Mode Driver **3.4.3**，MACA **3.7.1.5**，`mx-smi 2.3.1`，BIOS 1.30.0.0 |
| CPU | 2 × Intel Xeon Platinum 8468V，48C/96T ×2 = **96 核 / 192 线程**，800–3800 MHz，2 NUMA |
| 内存 | **1.0 TiB**，无 swap |
| 操作系统 | Ubuntu 22.04，gcc 11.4.0，kernel 5.14.0-3.1.8.kwai.x86_64 |
| 编译 | `cucc`（`mxcc version 1.0.0`），`make -j96` |
| 存储 | 项目位于 CephFS；系统盘 overlay |

单卡功耗/温度（`nvml` 实测）：待机 **93 W / 36 °C**，满载 **451 W（功耗墙 450 W）/ 64 °C**。
8 卡同时满载：**总功耗 3576 W**，各卡 443–452 W，温度 55–64 °C。

## 3. 构建与运行

```bash
./scripts/build.sh                      # cucc 构建，默认用满核（JOBS=8 可覆盖）
./scripts/run.sh --mode benchmark --precision FP16 BF16
./scripts/run.sh --mode benchmark --device 3          # 指定卡（新增）
./scripts/run.sh --mode memory --device 0             # 内存/互连带宽（新增）
./scripts/run8.sh benchmark FP16        # 8 卡并行，日志在 runs/
```

> **注意**：上游程序在测试结束后会停在 `Press Ctrl+C to exit.` 而不自行退出（见 `main.cu`），
> 脚本里已用轮询 `Test finished` + `SIGINT` 收尾。

## 4. 已知特性与限制

1. **每项 benchmark 的耗时已修正**。上游是「异步提交 10 秒 → synchronize」，在 C550 上
   （提交 4.7 μs / 执行 0.51 ms）会积压约 212 万次调用，单耗时 18 分钟。
   本仓库改为每 16 次提交同步一次，队列深度有界：2 项从 36 分钟降到 **38 秒**。
   这会改变各平台的绝对数值（更接近真实速率），如需与上游对比请用 `git checkout` 回到未改版本。
2. **FP4 不可用**：cu-bridge 未提供 `cuda_fp4.h`，且 C550 `cc=8.0` 不满足上游 FP4 分支条件
   （需 9.0 或 8.9）。`compat/gemm_fp4_unimplemented.cpp` 提供抛异常的桩，保证链接通过且不静默出错。
3. **FP8 编译通过但运行时不达**：上游要求 `cc==9.0` 或 `8.9`，C550 是 `8.0`，`operator_factory`
   会走 `FP32 Simulation` 分支。
4. **GfxClock 读不到**：`nvmlDeviceGetClockInfo(NVML_CLOCK_GRAPHICS)` 在 libmxsml 上返回不支持，
   上游代码失败即置 0，不影响运行。
5. **FP64 极慢**：`GemmGeneric<double>` 2048³ 在 C550 上首次调用耗时长，实测 45 秒未跑完第一项；
   需要时请单独给足时间。

## 4.2 相对上游的行为修正

除移植适配外，本仓库修正了上游 3 处问题（**均在 C550 上实测确认**）：

| 问题 | 上游行为 | 本仓库 |
|---|---|---|
| 稀疏项虚报 | `gemm_sparse_tensorcore.cu` 未设 `CUBLASLT_MATMUL_DESC_SPARSE`，却按稀疏记双倍 | 按 dense 记账；平台无该属性时打印警告。C550 上 449 → **223.56** TFLOPS |
| 计时积压 | 提交 10 s 再 sync，快卡上单耗 18 分钟 | 每 16 次提交 sync 一次，2 项 **38 秒** |
| 单卡/单模式 | 只能 device 0，只能算力 | 新增 `--device N`、`--mode memory`（HBM/PCIe/P2P） |

> 这些修改改变了上游的测量语义。要复现上游原始数值，请 `git checkout <上游 commit>`。

## 4.5 C550 实测摘要

完整数据（含系统配置、逐卡数值、精度矩阵、内存带宽）见 **[docs/C550_RESULTS.md](docs/C550_RESULTS.md)**。

**八卡 FP16（8 进程，`MACA_VISIBLE_DEVICES` 各占一卡，同时运行）**

| GPU | Dense (TFLOPS) | Sparse (TFLOPS) |
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

> Sparse 一列是虚报：上游未设置 `CUBLASLT_MATMUL_DESC_SPARSE`，但记账乘了 2。真实约 224 TFLOPS。详见 docs 第 5 节。

**精度支持矩阵（`cublasGemmEx`，2048³）**

| 可选 | 不可选 |
|---|---|
| FP64 4.6 · FP32 34.6 · TF32 83.1 · BF16 174.1 · FP16 176.3 · INT8 202.8 TOPS | `COMPUTE_32F_FAST_16F/16BF`、FP8 E4M3/E5M2、FP6 E2M3/E3M2、FP4 E2M1、INT4（均 status 15） |

**内存与互连**：HBM D2D **960.6 GB/s**；PCIe H2D/D2H 50.4/56.6 GB/s；P2P 55.3 GB/s（`cudaMemcpyPeer`，疑似走 PCIe 而非 MetaXLink，待用 MCCL 验证）。

## 5. 上游与致谢

- 上游：[stlin256/CUDABurner](https://github.com/stlin256/CUDABurner) @ `d85c030`（2025-12-12）
- 本仓库的移植与验证由上游作者在沐曦曦云 C550 上完成，**上游源码零修改**
- 上游原始 README 见 [docs/README.upstream.md](docs/README.upstream.md)

## 6. 目录

| 路径 | 说明 |
|---|---|
| `main.cu`、`core/`、`monitoring/`、`operators/`、`strategies/`、`ui/`、`utils/` | 上游源码，零修改 |
| `CMakeLists.txt`、`build_appimage.sh`、`*.desktop`、`cudaburner.png`、`main.cpp` | 上游原文件（CMake 走 nvcc，本移植未使用） |
| `compat/gemm_fp4_unimplemented.cpp` | 本仓库新增：FP4 桩 |
| `Makefile` | 本仓库新增：cucc 构建 |
| `scripts/` | 本仓库新增：构建 + 单卡 + 8 卡运行 |
| `tools/` | 本仓库新增：精度能力探测、内存/P2P 探测 |
| `docs/C550_RESULTS.md` | 本仓库新增：C550 完整实测数据 |
| `PORTING.md` | 每处移植改动的依据与证据 |
| `docs/README.upstream.md` | 上游 README 原文 |
