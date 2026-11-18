#include "operators/operator_factory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>


#include "operators/gemm_generic.hpp"
#include "operators/gemm_tensorcore.hpp"
#include "operators/gemm_sparse_tensorcore.hpp"
#include "operators/gemm_fp8_tensorcore.hpp"
#include "operators/gemm_fp4_tensorcore.hpp"

namespace {

// --- 矩阵尺寸选择 ---------------------------------------------------------
// 上游硬编码：FP64=2048, INT8=8192, 其余=4096。
// 实测（docs/C550_RESULTS.md 第 7 节，MetaX C550）显示部分精度在更大尺寸下
// 吞吐显著更高，尤以 TF32 为甚（4096: 106.3 -> 8192: 129.4 TFLOPS, +22%）。
//
// 类别：**实测标定值**（非源码推导）。出处为 tools/probe_size.cu 的扫描结果。
// 换平台/换库版本必须重测，不得直接沿用。
int upstream_dim(Precision p) {
    if (p == Precision::FP64) return 2048;
    if (p == Precision::INT8) return 8192;
    return 4096;
}

int tuned_dim(Precision p) {
    switch (p) {
        case Precision::FP64: return 2048;   // 2048 后已饱和
        case Precision::FP32: return 8192;   // 36.3 vs 4096 的 35.6
        case Precision::TF32: return 8192;   // 129.4 vs 4096 的 106.3 (+22%)
        case Precision::BF16: return 8192;   // 274.2; 16384 仅 +1.3% 但慢一倍
        case Precision::FP16: return 8192;   // 274.9; 16384 为 278.0
        case Precision::INT8: return 16384;  // 397.6 vs 8192 的 378.9 (+5%)
        default: return 4096;
    }
}

int resolve_dim(Precision p) {
    const char* env = std::getenv("MXBURNER_DIM");
    if (env == nullptr) return tuned_dim(p);
    if (std::strcmp(env, "upstream") == 0) return upstream_dim(p);
    if (std::strcmp(env, "tuned") == 0) return tuned_dim(p);
    int d = std::atoi(env);
    return d > 0 ? d : tuned_dim(p);
}

}  // namespace

namespace OperatorFactory {

std::unique_ptr<BaseOperator> create_gemm_operator(
    const OperatorDescriptor& desc, const GpuProperties& props, std::string& out_notes, int dim_override) {

    const int dim = dim_override > 0 ? dim_override : resolve_dim(desc.precision);
    int m = dim, n = dim, k = dim;

    if (desc.sparsity == Sparsity::SPARSE) {
        if (props.cc_major >= 9) {
            if (desc.precision == Precision::FP8) {
                out_notes = "Sparse Tensor Core";
                #if CUDA_VERSION >= 11080
                    return std::make_unique<GemmFp8TensorCore>(desc, m, n, k);
                #else
                    out_notes = "FP8 requires CUDA Toolkit 11.8+";
                    return nullptr;
                #endif
            }
        }
        if (props.cc_major >= 8) {
            if (desc.precision == Precision::FP16 || desc.precision == Precision::INT8) {
                out_notes = "Sparse Tensor Core";
                return std::make_unique<GemmSparseTensorCore>(desc, m, n, k);
            }
        }
    }

    switch (desc.precision) {
        case Precision::FP64:
            if (props.cc_major >= 6) { 
                out_notes = "CUDA Core (Native)";
                return std::make_unique<GemmGeneric<double>>(desc, m, n, k, true);
            }
            break;

        case Precision::FP32:
            out_notes = "CUDA Core (TC Disabled)";
            return std::make_unique<GemmGeneric<float>>(desc, m, n, k, true);

        case Precision::TF32:
            if (props.cc_major >= 8) {
                out_notes = "Tensor Core (Default FP32)";
                return std::make_unique<GemmTensorCore>(desc, props, m, n, k);
            }
            break;
        
        case Precision::FP16:
        case Precision::BF16:
        case Precision::INT8:
            if (props.cc_major >= 7) {
                 out_notes = "Tensor Core";
                 return std::make_unique<GemmTensorCore>(desc, props, m, n, k);
            }
            break;
            
        case Precision::FP8:
            {
                // Official FP8 support is on Hopper (sm_9x) and Ada (sm_89).
                bool is_supported = (props.cc_major == 9) || (props.cc_major == 8 && props.cc_minor == 9);
                #if CUDA_VERSION < 11080
                    is_supported = false;
                #endif

                if (is_supported) {
                    out_notes = "Tensor Core";
                    return std::make_unique<GemmFp8TensorCore>(desc, m, n, k);
                } else {
                    out_notes = "FP32 Simulation (Unsupported)";
                    return std::make_unique<GemmGeneric<float>>(desc, m, n, k, true);
                }
            }
            
        case Precision::FP4:
            {
                // Assume FP4 has the same support as FP8 for now.
                bool is_supported = (props.cc_major == 9) || (props.cc_major == 8 && props.cc_minor == 9);
                #if CUDA_VERSION < 12030
                    is_supported = false;
                #endif

                if (is_supported) {
                    out_notes = "Tensor Core";
                    return std::make_unique<GemmFp4TensorCore>(desc, m, n, k);
                } else {
                    out_notes = "FP32 Simulation (Unsupported)";
                    return std::make_unique<GemmGeneric<float>>(desc, m, n, k, true);
                }
            }
    }

    if(out_notes.empty()) out_notes = "Not Supported";
    return nullptr;
}

}
