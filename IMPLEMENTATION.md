# 实现说明：沐曦 MACA 适配

记录 mx-burner 在沐曦曦云 C550 上跑通所依赖的每个非显然决定，以及判断依据。
换平台或升级 MACA 后，这份文档是最先要复查的东西。

实测环境：C550 ×8，Driver 3.4.3，MACA 3.7.1.5，cu-bridge（`mxcc 1.0.0`）。

---

## 1. 沐曦的 CUDA 兼容层是怎么组织的

理解结构才能理解后面为什么加那些编译参数。

```
/opt/maca/tools/cu-bridge/
├── bin/cucc              # 编译器入口（实际转调 mxcc，LLVM 后端）
├── bin/nvcc -> cucc      # 兼容符号链接
├── include/
│   ├── cuda_runtime.h    # 12 行，转发到 __cuda_runtime.h + cuda_runtime_api.h
│   ├── cublas_v2.h       # 10 行，转发到 bridge/blas/cublas_v2_wrapper.h
│   ├── cublasLt.h        # 8 行，转发到 bridge/blas/cublasLt_wrapper.h
│   ├── cuda_fp16.h       # 6 行，转发到 cuda_fp16.hpp
│   ├── cuda_bf16.h       # 35 行，把 __nv_bfloat16 typedef 成 maca_bfloat16
│   ├── cuda_fp8.h        # 6 行
│   ├── nvml.h            # 14 行，转发到 bridge/sml/nvml_wrapper.h
│   └── library_types.h   # CUDA_R_* → MACA_R_*（34 条别名）
├── lib/libcuda.so -> ../../../lib/libsymbol_cu.so
└── cmake_module/
```

要点：这些头都是**转发头**，真正实现在：

- `cublas` → `libmcblas.so` / `libmcblasLt.so`（`/opt/maca/lib`）
- `cudart` → `libcudart`
- `nvml` → `libmxsml.so`（`/opt/mxdriver/lib`），头为 `MxSml.h` / `MxSmlExtension.h`（`/opt/maca/include/mxsml`）

所以链接参数是：

```make
LIBS = -L/opt/maca/lib -L/opt/mxdriver/lib \
       -lcudart -lmcblas -lmcblasLt -lmxsml -lpthread
```

`cucc` 同时能编 `.cu` 和 `.cpp`，用它编全部 TU，避免 g++ 与 cucc 的宏/ABI 差异。

---

## 2. 编译期适配（三处）

CUDA 路径的算子与测量代码没有为沐曦改动，适配全部在编译期完成。

### 2.1 `-DCUDA_VERSION=12030`

**为什么需要**：`operators/gemm_fp8_tensorcore.hpp:7` 与 `gemm_fp4_tensorcore.hpp:7` 把
`GemmFp8TensorCore` / `GemmFp4TensorCore` 的**类声明**包在 `#if CUDA_VERSION >= 11080 / 12030` 里，
而 `operators/operator_factory.cu:77` 与 `:94` **无条件**引用这两个类：

```cpp
return std::make_unique<GemmFp8TensorCore>(desc, m, n, k);   // :77
return std::make_unique<GemmFp4TensorCore>(desc, m, n, k);   // :94
```

不定义或定低了直接编译失败：

```
operators/operator_factory.cu:77:45: error: unknown type name 'GemmFp8TensorCore'
operators/operator_factory.cu:94:45: error: unknown type name 'GemmFp4TensorCore'
```

**为什么是 12030**：这是让两个类声明都成立的最小值。

**为什么不影响行为**：`CUDA_VERSION` 只出现在三处条件编译里，不参与任何运行时判断。
实际走不走 FP8/FP4 由 `cudaGetDeviceProperties` 的 `cc_major`/`cc_minor` 决定（C550 是 8.0）。

### 2.2 `-DNVML_DEVICE_NAME_BUFFER_SIZE=NVML_DEVICE_NAME_V2_BUFFER_SIZE`

**为什么需要**：`bridge/sml/nvml_wrapper.h` 只导出了 V2 版常量：

```c
:15  #define NVML_DEVICE_NAME_V2_BUFFER_SIZE MXSMLEX_NAME_V2_BUFFER_SIZE
:17  #define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE MXSMLEX_DRIVER_VERSION_BUFFER_SIZE
```

而 `monitoring/monitor.hpp` 的结构体用的是旧名：

```cpp
char name[NVML_DEVICE_NAME_BUFFER_SIZE];
```

编译报 `use of undeclared identifier 'NVML_DEVICE_NAME_BUFFER_SIZE'`。
NVML 语义里这两个常量指向同一容量（96），用 `-D` 补别名等价于 CUDA 头里本就存在的定义。

### 2.3 `-I/opt/maca/include/mxsml -include MxSml.h`

**为什么需要**：throttle 相关常量经过三层别名：

1. `monitoring/monitor.cpp:86-100` 用 `nvmlClocksThrottleReasonGpuIdle` / `_SwPowerCap` / `_HwSlowdown` / `_SyncBoost`
2. `nvml_wrapper.h:21` 映射到 Ex 命名：`#define nvmlClocksThrottleReasonGpuIdle mxsmlExClocksThrottleReasonGpuIdle`
3. `nvml_wrapper.h:8` 只 include 了 `MxSmlExtension.h`，而 `mxsmlEx*` 最终指向的底层宏
   `mxsmlClocksThrottleReasonIdle` 定义在 `/opt/maca/include/mxsml/MxSml.h:2239`，**没有被包含进来**

编译报 `use of undeclared identifier 'mxsmlClocksThrottleReasonIdle'`。

解法是把底层宏提前引入，而不是自己写一遍字面量（那属于硬编码常量）：

```make
INC     = -I/opt/maca/include/mxsml ...
FORCEIN = -include MxSml.h
```

---

## 3. FP4 桩

**为什么不能编译原文件**：`operators/gemm_fp4_tensorcore.cu:5` include `<cuda_fp4.h>`，
cu-bridge 的 include 目录里有 `cuda_fp8.h`，**没有 `cuda_fp4.h`**（该头随 CUDA 12.3 提供）。

**为什么运行时不可达**：`operator_factory.cu` 的 FP4 分支判据是

```cpp
bool is_supported = (props.cc_major == 9) || (props.cc_major == 8 && props.cc_minor == 9);
```

C550 实测 `cc = 8.0`，条件为 false，工厂返回 `GemmGeneric<float>`（`FP32 Simulation`）。
`GemmFp4TensorCore` 永不被构造。

**做法**：`compat/gemm_fp4_unimplemented.cpp` 提供该类 4 个非 inline 符号（构造、析构、`execute`、
`get_gflops_or_gops`），构造即抛 `std::runtime_error`。平台假设一旦变化会明确失败，而不是静默给错数。

**没有伪造 `cuda_fp4.h`**：伪造会让人误以为本平台支持 FP4。宁可让它显式不可用。

---

## 4. 构建体系

`CMakeLists.txt` 是 CUDA 路径的构建文件，三处假设在沐曦上不成立：

| 假设 | 在沐曦上 |
|---|---|
| `project(... LANGUAGES CXX CUDA)` | CMake 需要 nvcc；`cucc --version` 实际打印 `mxcc version 1.0.0` |
| `CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 120` | 这些是 NVIDIA sm_xx；cucc 用 `-offload-arch=<value>`，实测不指定即可编出可运行的二进制 |
| `find_library(NVML_LIB nvidia-ml)` | 没有 `libnvidia-ml`，对应物是 `/opt/mxdriver/lib/libmxsml.so` |

因此沐曦路径用一份 `Makefile` 直接调 cucc。两者并存，按平台选用。

---

## 5. 运行时矩阵尺寸探测

**为什么要探测**：矩阵边长直接决定吞吐，且最优点随平台与库版本漂移。C550 实测：

| dim | FP64 | FP32 | TF32 | BF16 | FP16 | INT8 |
|---|---|---|---|---|---|---|
| 2048 | 4.6 | 34.5 | 83.0 | 164.5 | 175.1 | 200.0 |
| 4096 | 4.7 | 35.6 | 106.3 | 267.8 | 268.4 | 368.2 |
| 8192 | 4.8 | 36.3 | **129.4** | 274.2 | 274.9 | 378.9 |
| 16384 | – | 35.8 | 120.5 | **277.7** | **278.0** | **397.6** |

TF32 在 4096 → 8192 涨 22%，但在 16384 又掉到 120.5；FP64 在 2048 已饱和。
固定的 4096 对 TF32 明显偏低。

**实现**：`BenchmarkStrategy::autotune_dim` 对每个 `(precision, sparsity)` 在
`{2048, 4096, 8192, 16384}` 上短测（FP64 上限 4096），取最高者缓存；`OperatorFactory` 增加
`dim` 参数；`notes` 追加 `@<dim>` 标明本次边长。

**探测必须与正式测试同构**：每 8 次提交同步一次、跑满 1 秒。
初版用「固定 8 次迭代」，TF32 误选 16384——短测 120 TFLOPS，跑满 10 秒只有 60.4。

**类别标注**：`operator_factory.cu` 里的 `tuned_dim()` 是**实测标定值**（非源码推导），
出处是 `tools/probe_size.cu` 的扫描结果，换平台必须重测。默认走 autotune，
只有显式 `--size legacy` 才用固定边长。

---

## 6. 稀疏记账判定

`gemm_sparse_tensorcore.cu` 只设置了转置属性：

```cpp
cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &op_trans_a, sizeof(op_trans_a));
```

没有 `CUBLASLT_MATMUL_DESC_SPARSE`，送入的也不是 2:4 压缩矩阵，所以实际执行的是 dense 的转置 GEMM。

现在的记账方式：按 dense 计（不再乘 2），并在平台缺该属性时打印一行警告。
C550 上该项从虚报的 449 回落到 **223.68 TFLOPS**（dense 是 269.63），符合转置布局的预期。

沐曦侧确认：`cublasLt_to_mcblasLt_adaptor.h` 只有 `TRANSA` 等映射，无 `SPARSE`；
`/opt/maca/include/mcblasLt.h` 里也没有该属性；系统有独立的 `mcsparse` 库，但那是 cuSPARSE 对应物。

CUDA 路径若要用真稀疏，需要：生成 2:4 压缩矩阵 + 设置 `CUBLASLT_MATMUL_DESC_SPARSE` + 按压缩后的
存储计算操作数。这部分尚未实现。

---

## 7. 验证矩阵

| 能力 | 测法 | 结果 |
|---|---|---|
| CUDA 运行时 | 最小 vectorAdd + `cudaGetDeviceProperties` | `MetaX C550 cc=8.0 sm=104` |
| cuBLAS FP32 | `cublasSgemm` 1024³，A=B=1 | `C[0] = 1024.0`（= N） |
| cuBLAS TF32 | `cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)` | `status = 0` |
| cuBLAS FP16 | `cublasGemmEx` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | `status = 0` |
| cuBLAS INT8 | `cublasGemmEx` + `CUBLAS_COMPUTE_32I` | `status = 0` |
| 单次 GEMM | 4096³ FP16，预热后 20 次 | **0.51 ms → 270.3 TFLOPS** |
| 提交开销 | 1000 次 `cublasGemmEx` 不 sync | 4.7 μs/次 |
| NVML | 10 个 API 逐个调用 | 见下表 |
| 内存 | `cudaMemcpy` / `cudaMemcpyPeer` | HBM 960.6 GB/s，P2P 55.3 GB/s |

NVML 逐项：

| API | 结果 |
|---|---|
| `nvmlInit` | 0 |
| `nvmlDeviceGetHandleByIndex` / `GetName` | 0 / `MXC550` |
| `nvmlSystemGetDriverVersion` | `3.4.3` |
| `nvmlDeviceGetTemperature(NVML_TEMPERATURE_GPU)` | 36 °C 待机 → 64 °C 满载 |
| `nvmlDeviceGetPowerUsage` / `GetEnforcedPowerLimit` | 93 W 待机 → 451 W；上限 450 W |
| `nvmlDeviceGetUtilizationRates` | 0 % 待机 → 98 % 满载 |
| `nvmlDeviceGetClockInfo(NVML_CLOCK_MEM)` | 1800 MHz |
| `nvmlDeviceGetClockInfo(NVML_CLOCK_GRAPHICS)` | **status 3（不支持）** → 代码置 0 |
| `nvmlDeviceGetCurrentClocksThrottleReasons` | 待机 `1`（Idle）；满载 `[Max Performance]` |

270.3 TFLOPS 与同机 PyTorch 直测 274 TFLOP/s 一致 → 真 Tensor Core，非模拟。

---

## 8. 未确认

| 项 | 说明 | 已试过 |
|---|---|---|
| 稀疏 GEMM 真实带宽 | 未实现 2:4 压缩矩阵，当前条目只测 dense 转置 | 查 cu-bridge/mcblasLt 属性、查 `mcsparse` |
| FP8 运行时路径 | `cc=8.0` 使工厂走 FP32 模拟分支，从未触达真实 FP8 | 按 cc 条件推断 |
| `CUDA_R_4F_E2M1` 是否被 mcblas 支持 | `library_types.h:39` 有别名，但 GEMM 拒绝 | `cublasGemmEx` 实测 status 15 |
| MetaXLink 真实带宽 | `cudaMemcpyPeer` 只有 55 GB/s，疑似走 PCIe | 需 MCCL（`libmccl`）all-to-all 验证，未做 |
| GfxClock 替代接口 | `NVML_CLOCK_GRAPHICS` 不支持 | 未查 `MxSml.h` 是否有专用频率接口 |
