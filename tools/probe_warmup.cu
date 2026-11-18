// 预热敏感性：冷启动后做 N 次预热，再测 secs 秒稳态吞吐。
// 用法: ./probe_warmup <warmup_iters> <measure_seconds>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

int main(int argc, char** argv) {
    const int warmup = argc > 1 ? atoi(argv[1]) : 10;
    const double secs = argc > 2 ? atof(argv[2]) : 1.0;
    const int M = 4096;

    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { printf("create fail\n"); return 1; }
    float alpha = 1.f, beta = 0.f;
    void *A = nullptr, *B = nullptr, *C = nullptr;
    size_t sh = (size_t)M * M * 2;
    if (cudaMalloc(&A, sh) || cudaMalloc(&B, sh) || cudaMalloc(&C, sh)) { printf("malloc fail\n"); return 1; }

    auto gemm = [&] {
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, M, M, &alpha,
                     A, CUDA_R_16F, M, B, CUDA_R_16F, M, &beta, C, CUDA_R_16F, M,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };

    // 预热：N 次连续提交后 sync 一次（模拟上游 for(i<10) 的形态）
    for (int i = 0; i < warmup; ++i) gemm();
    cudaDeviceSynchronize();

    // 测量：每 16 次 sync 一次，避免队列积压
    long long iters = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    double elapsed = 0;
    while (elapsed < secs) {
        for (int k = 0; k < 16; ++k) { gemm(); iters++; }
        cudaDeviceSynchronize();
        elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    }
    cudaDeviceSynchronize();
    double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    double tflops = iters * 2.0 * M * M * M / dt / 1e12;
    printf("warmup=%-5d warmup_ms=%-8.2f iters=%-6lld wall=%.3fs  %.1f TFLOPS\n",
           warmup, warmup * 0.51, iters, dt, tflops);
    cublasDestroy(h);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
