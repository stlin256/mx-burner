# 移植说明（CUDABurner → 沐曦 MACA / 曦云 C550）

上游：`stlin256/CUDABurner` @ `d85c0303271699891e86f4548e3fd5b6a813c2e9`（2025-12-12）
目标：沐曦曦云 C550 ×8，Driver 3.4.3，MACA 3.7.1.5

**上游源码零修改。** 所有适配都在编译期完成，共 3 项动作 + 1 项构建体系替换。

---

## 1. 四项动作一览

| # | 动作 | 类别 | 放在哪 |
|---|---|---|---|
| 1 | `-DCUDA_VERSION=12030` | 上游源码的条件编译约束 | `Makefile` 的 `DEFS` |
| 2 | `-DNVML_DEVICE_NAME_BUFFER_SIZE=NVML_DEVICE_NAME_V2_BUFFER_SIZE` | cu-bridge 缺别名 | `Makefile` 的 `DEFS` |
| 3 | `-I/opt/maca/include/mxsml -include MxSml.h` | cu-bridge 别名链末端未自动包含 | `Makefile` 的 `INC` / `FORCEIN` |
| 4 | `compat/gemm_fp4_unimplemented.cpp` 替代 `operators/gemm_fp4_tensorcore.cu` | **自造替代实现（需登记）** | `compat/`，`Makefile` 的 `SRCS` |

外加：构建体系从上游 CMake（依赖 `project(... CUDA)` 的 nvcc 探测 + `find_library(nvidia-ml)`）换成
一份直接调 `cucc` 的 `Makefile`。上游 `CMakeLists.txt` 原样保留在仓库，仅作参考、不参与构建。

---

## 2. 动作 1：`CUDA_VERSION=12030`

**依据（上游源码，条件编译）：**

- `operators/gemm_fp8_tensorcore.hpp:7` —— `#if CUDA_VERSION >= 11080` **包住了整个 `GemmFp8TensorCore` 类声明**
- `operators/gemm_fp4_tensorcore.hpp:7` —— `#if CUDA_VERSION >= 12030` 包住 `GemmFp4TensorCore` 类声明
- `operators/operator_factory.cu:77` 与 `:94` —— **无条件**出现
  `return std::make_unique<GemmFp8TensorCore>(...)` 和 `return std::make_unique<GemmFp4TensorCore>(...)`

即：类声明被 `#if` 挡住，但工厂函数无条件引用它们。若 `CUDA_VERSION` 低于 12030，编译直接失败：

```
operators/operator_factory.cu:77:45: error: unknown type name 'GemmFp8TensorCore'
operators/operator_factory.cu:94:45: error: unknown type name 'GemmFp4TensorCore'
```

上游 CMake 里 `CUDA_VERSION` 是从 `CMAKE_CUDA_COMPILER_VERSION` 推算的（`CMakeLists.txt` 中
`math(EXPR CUDA_VERSION_INT ...)`，取 `major*1000 + minor*10`）。cucc 不是 nvcc，不走那条路径，
因此需要显式给出。

**取 12030 而不是更高**：12030 是让两个类声明都成立的最小值。取更高没有额外收益。

**这不是"假装有 CUDA 12.3"**：`CUDA_VERSION` 在上游只用于三处条件编译
（FP8 类声明 ≥11080、FP4 类声明 ≥12030、`operator_factory` 里把 `is_supported` 置 false），
不参与任何运行时行为。运行时是否真走 FP8/FP4 由 `cudaGetDeviceProperties` 的 `cc_major/cc_minor` 决定。

---

## 3. 动作 2：补 `NVML_DEVICE_NAME_BUFFER_SIZE`

**依据（cu-bridge 头文件）：**

`/opt/maca/tools/cu-bridge/include/bridge/sml/nvml_wrapper.h` 里只有 V2 版常量：

```c
:15  #define NVML_DEVICE_NAME_V2_BUFFER_SIZE MXSMLEX_NAME_V2_BUFFER_SIZE
:17  #define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE MXSMLEX_DRIVER_VERSION_BUFFER_SIZE
```

而上游 `monitoring/monitor.hpp` 的 `GpuState` 结构体用的是旧名：

```cpp
char name[NVML_DEVICE_NAME_BUFFER_SIZE];
```

`monitoring/monitor.cpp` 里 `nvmlDeviceGetName(device_handle_, current_state_.name, NVML_DEVICE_NAME_BUFFER_SIZE)`
同样使用旧名。编译报：

```
monitoring/monitor.cpp:9:13: error: use of undeclared identifier 'NVML_DEVICE_NAME_BUFFER_SIZE'
```

NVML 官方语义中 `NVML_DEVICE_NAME_BUFFER_SIZE`(96) 与 `NVML_DEVICE_NAME_V2_BUFFER_SIZE` 指向同一容量，
cu-bridge 只导出了后者。用 `-D` 做别名，等价于 CUDA 头里本应存在的定义。

---

## 4. 动作 3：强制包含 `MxSml.h`

**现象：**

```
monitoring/monitor.cpp:86:65: error: use of undeclared identifier 'mxsmlClocksThrottleReasonIdle'
monitoring/monitor.cpp:86:100: error: use of undeclared identifier 'mxsmlClocksThrottleReasonOther'
monitoring/monitor.cpp:90:27: error: use of undeclared identifier 'mxsmlClocksThrottleReasonOverPower'
```

**链条：**

1. 上游 `monitor.cpp:86-100` 使用 `nvmlClocksThrottleReasonGpuIdle` / `_SwPowerCap` / `_HwSlowdown` / `_SyncBoost`
2. `nvml_wrapper.h:21` 把它们映射到 Ex 命名：
   ```c
   :18  #define nvmlClocksThrottleReasonAll mxsmlExClocksThrottleReasonAll
   :21  #define nvmlClocksThrottleReasonGpuIdle mxsmlExClocksThrottleReasonGpuIdle
   :26  #define nvmlClocksThrottleReasonSwPowerCap mxsmlExClocksThrottleReasonSwPowerCap
   ```
3. `nvml_wrapper.h:8` 只 `#include "MxSmlExtension.h"`
4. `MxSmlExtension.h` 内部再引用 `mxsmlClocksThrottleReasonIdle` 等**底层宏**，它们定义在
   `/opt/maca/include/mxsml/MxSml.h:2239` 起：
   ```c
   :2239 #define mxsmlClocksThrottleReasonIdle        0x0000000000000001LL
   :2250 #define mxsmlClocksThrottleReasonOverPower   0x0000000000000004LL
   :2295 #define mxsmlClocksThrottleReasonOther       0x0000000080000000LL
   ```
5. 但 `MxSml.h` 没有被 wrapper 链包含进来 → 底层宏未定义

因此用 `-include MxSml.h` 把底层宏提前引入（同时把 `/opt/maca/include/mxsml` 加进 include path）。
这属于 cu-bridge 头文件自身的包含顺序问题，不是上游代码的问题。

> 注：不加这个，也可以自己写一个 compat 头把 4 个宏 `#define` 到字面量——但那就是硬编码常量，
> 不如直接包含沐曦自己的定义。

---

## 5. 动作 4：FP4 桩（自造替代实现，必须登记）

**为什么不能编译原文件：**

- `operators/gemm_fp4_tensorcore.cu:5` —— `#include <cuda_fp4.h>`
- `/opt/maca/tools/cu-bridge/include/` 里有 `cuda_fp8.h`，**没有 `cuda_fp4.h`**（FP4 头随 CUDA 12.3 提供）

```
operators/gemm_fp4_tensorcore.cu:5:10: fatal error: 'cuda_fp4.h' file not found
```

**为什么运行时不可达：**

`operators/operator_factory.cu` 的 FP4 分支：

```cpp
bool is_supported = (props.cc_major == 9) || (props.cc_major == 8 && props.cc_minor == 9);
```

目标机实测 `cudaGetDeviceProperties` 返回 `cc = 8.0`（major=8, minor=0），条件为 false，
工厂会返回 `GemmGeneric<float>`（`FP32 Simulation`）。**`GemmFp4TensorCore` 永不被构造。**

**桩的做法：**

`compat/gemm_fp4_unimplemented.cpp` 提供该类 4 个非 inline 符号（构造、析构、`execute`、
`get_gflops_or_gops`），构造函数直接抛 `std::runtime_error`：
一旦平台假设变化（真的走到 FP4 分支），会明确失败，而不是静默返回错误结果。

**没有伪造 `cuda_fp4.h`**：刻意没有提供自制 FP4 头 —— 那会让人误以为本平台支持 FP4。
宁可让这个类显式不可用。

---

## 6. 构建体系替换

上游 `CMakeLists.txt` 的三处硬假设在 cu-bridge 下不成立：

| 上游写法 | 在沐曦上的问题 |
|---|---|
| `project(CUDABurner LANGUAGES CXX CUDA)` | CMake 需要 nvcc，`cucc` 不是 nvcc（`cucc --version` 实际打印 `mxcc version 1.0.0`） |
| `set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 120)` | 这些是 NVIDIA sm_xx；cucc 用 `-offload-arch=<value>`（例：`xcore1000`），实测默认不指定也能正常编译运行 |
| `find_library(NVML_LIB nvidia-ml)` | 沐曦没有 `libnvidia-ml`，对应物是 `/opt/mxdriver/lib/libmxsml.so` |
| `target_link_libraries(... cublas cublasLt cudart)` | 实际库名是 `libmcblas` / `libmcblasLt` / `libcudart`（随 cu-bridge 提供） |

因此改为一份 30 行的 `Makefile`，直接 `cucc` 编译每个 TU 再链接。
`cucc` 同时能编 `.cu` 和 `.cpp`，避免 g++ 与 cucc 的 ABI/宏差异。

链接参数：

```make
LIBS = -L/opt/maca/lib -L/opt/mxdriver/lib \
       -lcudart -lmcblas -lmcblasLt -lmxsml -lpthread
```

---

## 7. 实测验证矩阵

| 能力 | 测法 | 结果 |
|---|---|---|
| CUDA 运行时 | 最小 vectorAdd + `cudaGetDeviceProperties` | `name=MetaX C550 cc=8.0 sm=104` |
| cuBLAS FP32 | `cublasSgemm` 1024³，A=B=1 | `C[0] = 1024.0`（= N，数值正确） |
| cuBLAS TF32 | `cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)` | `status = 0` |
| cuBLAS FP16 | `cublasGemmEx(..., CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)` | `status = 0` |
| cuBLAS INT8 | `cublasGemmEx(..., CUDA_R_8I, CUBLAS_COMPUTE_32I)` | `status = 0`（支持） |
| 单次 GEMM 吞吐 | 4096³ FP16，预热 3 次后 20 次计时 | **0.51 ms → 270.3 TFLOPS** |
| CPU 侧提交开销 | 1000 次 `cublasGemmEx` 不 sync | 4.7 μs/call |
| NVML init | `nvmlInit` | `0` |
| 设备句柄/名称 | `nvmlDeviceGetHandleByIndex` / `GetName` | `0` / `MXC550` |
| 驱动版本 | `nvmlSystemGetDriverVersion` | `3.4.3` |
| 温度 | `nvmlDeviceGetTemperature(NVML_TEMPERATURE_GPU)` | `36 °C`（待机）→ `64 °C`（满载） |
| 功耗/上限 | `nvmlDeviceGetPowerUsage` / `GetEnforcedPowerLimit` | `93 W`（待机）→ `451 W` / 上限 `450 W` |
| 利用率 | `nvmlDeviceGetUtilizationRates` | `gpu 0%`（待机）→ `98%`（满载） |
| 显存频率 | `nvmlDeviceGetClockInfo(NVML_CLOCK_MEM)` | `1800 MHz` |
| 图形频率 | `nvmlDeviceGetClockInfo(NVML_CLOCK_GRAPHICS)` | **`status = 3`（不支持）** → 上游置 0 |
| throttle | `nvmlDeviceGetCurrentClocksThrottleReasons` | 待机 `1`（Idle）、满载 `[Max Performance]` |

---

## 8. 未确认 / 未验证

| 项 | 说明 | 检索/尝试过什么 |
|---|---|---|
| 稀疏 GEMM（`GemmSparseTensorCore`） | 编译链接通过，但 `cublasLtMatmul` 的稀疏路径未单独验证 | 上游用 `CUBLASLT_MATMUL_DESC_TRANSA` + `cublasLtMatmul`，未设置 Sparse 属性；C550 上是否真走稀疏单元未确认 |
| FP8 运行时路径 | 编译通过；运行时因 `cc=8.0` 走 `FP32 Simulation` 分支，未触达真实 FP8 | 上游条件 `cc==9 \|\| (cc==8 && cc_minor==9)` |
| `CUDA_R_4F_E2M1` 是否真被 mcblas 支持 | `library_types.h:39` 有 `#define CUDA_R_4F_E2M1 MACA_R_4F_E2M1`，但未实测 | — |
| GfxClock 是否有替代 API | libmxsml 对 `NVML_CLOCK_GRAPHICS` 返回不支持 | 未查 `MxSml.h` 是否有专用频率接口 |
| 8 卡 benchmark 全项数值 | 每项约 18 分钟（原因见 README 第 4 节），单轮跑完需数小时 | 见 `docs/C550_RESULTS.md` |
