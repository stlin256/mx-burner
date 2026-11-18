// 探测 MACA/cu-bridge 上 cublasGemmEx 支持哪些精度，并给出 2048^3 的实测吞吐
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <string>

struct Case { const char* name; cudaDataType_t dt; cudaDataType_t ct; cublasComputeType_t comp; double bytes_per_elem; };

int main() {
    const int M = 2048, N = 2048, K = 2048;
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { printf("cublasCreate failed\n"); return 1; }

    std::vector<Case> cases = {
        {"FP64",        CUDA_R_64F,     CUDA_R_64F, CUBLAS_COMPUTE_64F,          8.0},
        {"FP32",        CUDA_R_32F,     CUDA_R_32F, CUBLAS_COMPUTE_32F,          4.0},
        {"TF32",        CUDA_R_32F,     CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, 4.0},
        {"BF16",        CUDA_R_16BF,    CUDA_R_16BF, CUBLAS_COMPUTE_32F,          2.0},
        {"FP16",        CUDA_R_16F,     CUDA_R_16F, CUBLAS_COMPUTE_32F,          2.0},
        {"FP16_c32f",   CUDA_R_16F,     CUDA_R_32F, CUBLAS_COMPUTE_32F,          2.0},
        {"FP16_fast",   CUDA_R_16F,     CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_16F, 2.0},
        {"BF16_c32f",   CUDA_R_16BF,    CUDA_R_32F, CUBLAS_COMPUTE_32F,          2.0},
        {"BF16_fast",   CUDA_R_16BF,    CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_16BF,2.0},
        {"INT8",        CUDA_R_8I,      CUDA_R_32I, CUBLAS_COMPUTE_32I,          1.0},
        {"FP8_E4M3",    CUDA_R_8F_E4M3, CUDA_R_16F, CUBLAS_COMPUTE_32F,          1.0},
        {"FP8_E4M3_f32c",CUDA_R_8F_E4M3, CUDA_R_32F, CUBLAS_COMPUTE_32F,          1.0},
        {"FP8_E5M2",    CUDA_R_8F_E5M2, CUDA_R_16F, CUBLAS_COMPUTE_32F,          1.0},
        {"FP6_E2M3",    CUDA_R_6F_E2M3, CUDA_R_16F, CUBLAS_COMPUTE_32F,          0.75},
        {"FP6_E3M2",    CUDA_R_6F_E3M2, CUDA_R_16F, CUBLAS_COMPUTE_32F,          0.75},
        {"FP4_E2M1",    CUDA_R_4F_E2M1, CUDA_R_16F, CUBLAS_COMPUTE_32F,          0.5},
        {"INT4",        CUDA_R_4I,      CUDA_R_32I, CUBLAS_COMPUTE_32I,          0.5},
    };

    printf("%-15s %-8s %-10s %s\n", "precision", "status", "TFLOPS", "note");
    printf("---------------------------------------------------------------\n");

    float alpha = 1.f, beta = 0.f;
    int32_t ai = 1, bi = 0;

    for (auto& c : cases) {
        size_t szA = (size_t)(M * K * c.bytes_per_elem) + 64;
        size_t szB = (size_t)(K * N * c.bytes_per_elem) + 64;
        size_t cbytes = (c.ct == CUDA_R_32I) ? 4.0 : (c.ct == CUDA_R_32F || c.ct == CUDA_R_64F) ? (c.ct==CUDA_R_64F?8.0:4.0) : 2.0;
        size_t szC = (size_t)(M * N * cbytes) + 64;
        void *A = nullptr, *B = nullptr, *C = nullptr;
        if (cudaMalloc(&A, szA) || cudaMalloc(&B, szB) || cudaMalloc(&C, szC)) {
            printf("%-15s %-8s %-10s %s\n", c.name, "NOMEM", "-", "cudaMalloc failed");
            cudaFree(A); cudaFree(B); cudaFree(C);
            continue;
        }
        void* pa = (c.comp == CUBLAS_COMPUTE_32I) ? (void*)&ai : (void*)&alpha;
        void* pb = (c.comp == CUBLAS_COMPUTE_32I) ? (void*)&bi : (void*)&beta;
        cudaDataType_t ctype = c.ct;

        cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, pa,
                                         A, c.dt, M, B, c.dt, K, pb, C, ctype, M,
                                         c.comp, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("%-15s %-8d %-10s %s\n", c.name, (int)st, "-", "rejected by mcblas");
            cudaFree(A); cudaFree(B); cudaFree(C);
            continue;
        }
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("%-15s %-8s %-10s %s\n", c.name, "RUNERR", "-", cudaGetErrorString(e));
            cudaFree(A); cudaFree(B); cudaFree(C);
            continue;
        }
        // 计时
        const int IT = 10;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < IT; ++i)
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, pa, A, c.dt, M, B, c.dt, K, pb, C, ctype, M,
                         c.comp, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
        double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        double tflops = IT * 2.0 * M * N * K / dt / 1e12;
        printf("%-15s %-8s %-10.1f %s\n", c.name, "OK", tflops,
               c.comp == CUBLAS_COMPUTE_32I ? "TOPS" : "");
        cudaFree(A); cudaFree(B); cudaFree(C);
    }
    cublasDestroy(h);
    printf("PROBE_PREC_DONE\n");
    return 0;
}
