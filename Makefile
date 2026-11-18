# CUDABurner on MetaX C550 via cu-bridge (maca)
# CUDA 路径源码零修改；沐曦适配仅三项：
#   1) -D 补 wrapper 缺失的 NVML_DEVICE_NAME_BUFFER_SIZE 别名
#   2) -include MxSml.h 提供 mxsmlClocksThrottleReason* 真实宏（wrapper 别名链的末端）
#   3) FP4 单元以 compat/ 桩替代（cu-bridge 无 cuda_fp4.h，且 C550 cc=8.0 运行时不达）
CUCC   ?= /opt/maca/tools/cu-bridge/bin/cucc
CBB     = /opt/maca/tools/cu-bridge
INC     = -I$(CBB)/include -I/opt/maca/include/mxsml -I.
# CUDA_VERSION=12080：GemmFp8/GemmFp4 的类声明被包在
# #if CUDA_VERSION >= {11080,12080} 里，而 operator_factory.cu 会引用这两个类，
# 所以必须给足够高的版本。12080 同时也是 cuda_fp4.h 的真实门槛（12.8 起才有）。
DEFS    = -DCUDA_VERSION=12080 -DNVML_DEVICE_NAME_BUFFER_SIZE=NVML_DEVICE_NAME_V2_BUFFER_SIZE
LIBS    = -L/opt/maca/lib -L/opt/mxdriver/lib -lcudart -lmcblas -lmcblasLt -lmxsml -lpthread
FLAGS   = -O3 -std=c++17 -Wall -Wextra
# MxSml.h 提供 mxsmlClocksThrottleReason* 真实宏（wrapper 的别名链指向它）
FORCEIN = -include MxSml.h

SRCS = main.cu \
       monitoring/monitor.cpp \
       monitoring/memory_bench.cpp \
       operators/gemm_generic.cu \
       operators/gemm_tensorcore.cu \
       operators/vector_add.cu \
       operators/power_virus_kernel.cu \
       operators/gemm_sparse_tensorcore.cu \
       operators/gemm_fp8_tensorcore.cu \
       operators/operator_factory.cu \
       compat/gemm_fp4_unimplemented.cpp \
       strategies/stress_strategy.cpp \
       strategies/benchmark_strategy.cpp \
       ui/tui.cpp

OBJS = $(SRCS:.cu=.o)
OBJS := $(OBJS:.cpp=.o)

all: CUDABurner

%.o: %.cu
	@mkdir -p $(dir $@)
	$(CUCC) $(FLAGS) $(INC) $(DEFS) $(FORCEIN) -c $< -o $@

%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CUCC) $(FLAGS) $(INC) $(DEFS) $(FORCEIN) -c $< -o $@

CUDABurner: $(OBJS)
	$(CUCC) $(OBJS) -o $@ $(LIBS)

clean:
	rm -f $(OBJS) CUDABurner

.PHONY: all clean
