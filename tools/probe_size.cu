// 矩阵尺寸扫描：找出每种精度在 C550 上的吞吐饱和点。
// 固定边长 FP64=2048^3 / 其他=4096^3 / INT8=8192^3，未必是峰值。
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <chrono>

struct Prec {
    const char* name;
    cudaDataType_t dt;    // A/B
    cudaDataType_t ct;    // C
    cublasComputeType_t comp;
    double elem;          // A/B 字节
    double celem;         // C 字节
    int max_dim;          // 显存/时间上限
    const char* unit;
};

int main(int argc, char** argv) {
    Prec precs[] = {
        {"FP64", CUDA_R_64F,  CUDA_R_64F, CUBLAS_COMPUTE_64F,           8, 8, 8192,  "TFLOPS"},
        {"FP32", CUDA_R_32F,  CUDA_R_32F, CUBLAS_COMPUTE_32F,           4, 4, 16384, "TFLOPS"},
        {"TF32", CUDA_R_32F,  CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, 4, 4, 16384, "TFLOPS"},
        {"BF16", CUDA_R_16BF, CUDA_R_16BF,CUBLAS_COMPUTE_32F,           2, 2, 16384, "TFLOPS"},
        {"FP16", CUDA_R_16F,  CUDA_R_16F, CUBLAS_COMPUTE_32F,           2, 2, 16384, "TFLOPS"},
        {"INT8", CUDA_R_8I,   CUDA_R_32I, CUBLAS_COMPUTE_32I,           1, 4, 16384, "TOPS"},
    };
    const int dims[] = {1024, 2048, 4096, 8192, 16384};
    const int ndims = sizeof(dims) / sizeof(dims[0]);
    const int WARMUP = 3, ITERS = 8;

    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { printf("create fail\n"); return 1; }
    float alpha = 1.f, beta = 0.f;
    int32_t ai = 1, bi = 0;

    printf("%-6s", "dim");
    for (auto& p : precs) printf(" %14s", p.name);
    printf("\n");
    printf("-------------------------------------------------------------------------------------\n");

    for (int di = 0; di < ndims; ++di) {
        int M = dims[di];
        printf("%-6d", M);
        for (auto& p : precs) {
            if (M > p.max_dim) { printf(" %14s", "-"); continue; }
            size_t szA = (size_t)M * M * p.elem;
            size_t szB = szA;
            size_t szC = (size_t)M * M * p.celem;
            void *A = nullptr, *B = nullptr, *C = nullptr;
            if (cudaMalloc(&A, szA) || cudaMalloc(&B, szB) || cudaMalloc(&C, szC)) {
                printf(" %14s", "NOMEM");
                cudaFree(A); cudaFree(B); cudaFree(C);
                continue;
            }
            void* pa = (p.comp == CUBLAS_COMPUTE_32I) ? (void*)&ai : (void*)&alpha;
            void* pb = (p.comp == CUBLAS_COMPUTE_32I) ? (void*)&bi : (void*)&beta;
            auto gemm = [&] {
                cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, M, M, pa,
                             A, p.dt, M, B, p.dt, M, pb, C, p.ct, M,
                             p.comp, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            };
            for (int w = 0; w < WARMUP; ++w) gemm();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) { printf(" %14s", "RUNERR"); cudaGetLastError(); cudaFree(A); cudaFree(B); cudaFree(C); continue; }
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < ITERS; ++it) gemm();
            e = cudaDeviceSynchronize();
            if (e != cudaSuccess) { printf(" %14s", "RUNERR"); cudaGetLastError(); cudaFree(A); cudaFree(B); cudaFree(C); continue; }
            double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            double tflops = ITERS * 2.0 * M * M * M / dt / 1e12;
            printf(" %12.1f%s", tflops, p.unit[0] == 'T' && p.comp == CUBLAS_COMPUTE_32I ? "" : "");
            cudaFree(A); cudaFree(B); cudaFree(C);
        }
        printf("\n");
        fflush(stdout);
    }
    printf("-------------------------------------------------------------------------------------\n");
    printf("单位：FP64/FP32/TF32/BF16/FP16 为 TFLOPS，INT8 为 TOPS\n");
    printf("SIZE_SWEEP_DONE\n");
    cublasDestroy(h);
    return 0;
}
