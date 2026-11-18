// 兼容层文件（CUDA 路径不含此文件）
//
// 背景：
//   operators/gemm_fp4_tensorcore.cu 依赖 <cuda_fp4.h>，该头随 CUDA >= 12.3 提供。
//   MACA cu-bridge 3.7.1 的 include 目录里有 cuda_fp8.h，但没有 cuda_fp4.h。
//
// 本类在 C550 上运行时不可达：
//   operator_factory.cu 的 FP4 分支条件是
//     is_supported = (cc_major == 9) || (cc_major == 8 && cc_minor == 9)
//   目标机 cudaGetDeviceProperties 实测 cc = 8.0（major=8, minor=0），
//   因此 factory 会走 "FP32 Simulation (Unsupported)" 分支，永不构造 GemmFp4TensorCore。
//
// 此文件只提供链接所需的符号（虚表需要全部虚函数），构造时显式抛异常，
// 避免「以为在跑 FP4、其实静默出错」。
//
// 如果将来 cu-bridge 提供 cuda_fp4.h，请从 Makefile 移除 compat/gemm_fp4_unimplemented.cpp
// 并恢复 operators/gemm_fp4_tensorcore.cu。

#include "operators/gemm_fp4_tensorcore.hpp"
#include <stdexcept>

GemmFp4TensorCore::GemmFp4TensorCore(const OperatorDescriptor& desc, int m, int n, int k)
    : m_(m), n_(n), k_(k),
      cublaslt_handle_(nullptr), matmul_desc_(nullptr),
      descriptor_(desc), gflops_or_gops_(0.0) {
    throw std::runtime_error(
        "GemmFp4TensorCore: FP4 GEMM 在本平台不可用"
        "（MACA cu-bridge 未提供 cuda_fp4.h；该头要求 CUDA >= 12.3）。"
        "C550 cc=8.0 不满足 FP4 所需的 cc 9.0 / 8.9。");
}

GemmFp4TensorCore::~GemmFp4TensorCore() {}

void GemmFp4TensorCore::execute(cudaStream_t) {}

double GemmFp4TensorCore::get_gflops_or_gops() const { return 0.0; }
