// 探测 C550 的内存子系统：HBM 拷贝带宽、P2P(MetaXLink) 带宽、H2D/D2H
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

static double gbps(size_t bytes, double sec) { return bytes / sec / 1e9; }

int main() {
    int ndev = 0; cudaGetDeviceCount(&ndev);
    printf("device_count=%d\n", ndev);

    // ---- HBM D2D copy（单卡） ----
    cudaSetDevice(0);
    size_t N = 256ull << 20;   // 256 MiB
    void *a, *b;
    if (cudaMalloc(&a, N) || cudaMalloc(&b, N)) { printf("malloc fail\n"); return 1; }
    cudaMemset(a, 1, N);
    for (int i = 0; i < 3; ++i) cudaMemcpy(b, a, N, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    const int IT = 10;
    for (int i = 0; i < IT; ++i) cudaMemcpy(b, a, N, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    // D2D 拷贝 = 读 + 写
    printf("HBM  D2D  copy: %8.1f GB/s  (%.2f ms / %zu MiB)\n", gbps(2.0 * N * IT, dt), dt / IT * 1e3, N >> 20);

    // ---- H2D / D2H（pin 内存） ----
    void* h; cudaMallocHost(&h, N);
    for (int i = 0; i < 2; ++i) cudaMemcpy(a, h, N, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i) cudaMemcpy(a, h, N, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    printf("PCIe H2D  copy: %8.1f GB/s  (pinned)\n", gbps((double)N * 5, dt));
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i) cudaMemcpy(h, a, N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    printf("PCIe D2H  copy: %8.1f GB/s  (pinned)\n", gbps((double)N * 5, dt));

    // ---- P2P / MetaXLink ----
    if (ndev >= 2) {
        int peer01 = 0, peer10 = 0;
        cudaDeviceCanAccessPeer(&peer01, 0, 1);
        cudaDeviceCanAccessPeer(&peer10, 1, 0);
        printf("p2p access: 0->1=%d  1->0=%d\n", peer01, peer10);
        void *d0 = nullptr, *d1 = nullptr;
        if (peer01) {
            cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);
            cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);
            cudaSetDevice(0); cudaMalloc(&d0, N);
            cudaSetDevice(1); cudaMalloc(&d1, N);
            cudaSetDevice(0); cudaMemset(d0, 1, N);
            for (int i = 0; i < 3; ++i) cudaMemcpyPeer(d1, 1, d0, 0, N);
            cudaDeviceSynchronize();
            t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 10; ++i) cudaMemcpyPeer(d1, 1, d0, 0, N);
            cudaDeviceSynchronize();
            dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            printf("P2P   0->1 copy: %8.1f GB/s  (MetaXLink, %.2f ms / %zu MiB)\n",
                   gbps((double)N * 10, dt), dt / 10 * 1e3, N >> 20);
        }
        // 跨 NUMA 的一对：0 -> 7
        if (ndev >= 8 && d0) {
            int p07 = 0; cudaDeviceCanAccessPeer(&p07, 0, 7);
            printf("p2p access: 0->7=%d\n", p07);
            if (p07) {
                cudaSetDevice(0); cudaDeviceEnablePeerAccess(7, 0);
                cudaSetDevice(7); cudaDeviceEnablePeerAccess(0, 0);
                void* d7; cudaSetDevice(7); cudaMalloc(&d7, N);
                cudaSetDevice(0);
                for (int i = 0; i < 3; ++i) cudaMemcpyPeer(d7, 7, d0, 0, N);
                cudaDeviceSynchronize();
                t0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 10; ++i) cudaMemcpyPeer(d7, 7, d0, 0, N);
                cudaDeviceSynchronize();
                dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
                printf("P2P   0->7 copy: %8.1f GB/s  (cross-NUMA)\n", gbps((double)N * 10, dt));
            }
        }
    }
    cudaFreeHost(h); cudaFree(a); cudaFree(b);
    printf("PROBE_MEM_DONE\n");
    return 0;
}
