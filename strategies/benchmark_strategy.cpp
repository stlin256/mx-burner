#include "strategies/benchmark_strategy.hpp"
#include "monitoring/monitor.hpp" // Include GpuMonitor header
#include "operators/operator_factory.hpp"
#include "utils/helpers.hpp"
#include <chrono>
#include <algorithm>
#include <map>

BenchmarkStrategy::BenchmarkStrategy(GpuMonitor& monitor, const GpuProperties& props, const std::vector<std::string>& selected_precisions)
    : monitor_(monitor), gpu_props_(props) {
    
    std::vector<OperatorDescriptor> all_tests = {
        {Precision::FP64, Sparsity::DENSE},
        {Precision::FP32, Sparsity::DENSE},
        {Precision::TF32, Sparsity::DENSE},
        {Precision::FP16, Sparsity::DENSE},
        {Precision::FP16, Sparsity::SPARSE},
        {Precision::BF16, Sparsity::DENSE},
        {Precision::INT8, Sparsity::DENSE},
        {Precision::INT8, Sparsity::SPARSE},
        // {Precision::FP8,  Sparsity::DENSE},
        // {Precision::FP8,  Sparsity::SPARSE},
        // {Precision::FP4,  Sparsity::DENSE},
    };

    if (selected_precisions.empty()) {
        tests_to_run_ = all_tests;
    } else {
        // Create a reverse map from string to Precision enum
        std::map<std::string, Precision> name_to_precision;
        for(const auto& pair : PRECISION_NAMES) {
            name_to_precision[pair.second] = pair.first;
        }

        for (const auto& selected : selected_precisions) {
            if (name_to_precision.count(selected)) {
                Precision p = name_to_precision.at(selected);
                // Find all tests (dense and sparse) matching this precision
                for(const auto& test : all_tests){
                    if(test.precision == p){
                        tests_to_run_.push_back(test);
                    }
                }
            } else {
                std::cerr << "Warning: Unknown precision '" << selected << "' ignored." << std::endl;
            }
        }
    }
}

const std::vector<BenchmarkResult>& BenchmarkStrategy::get_results() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return results_;
}

void BenchmarkStrategy::start() {
    stop_flag_ = false;
    worker_thread_ = std::thread(&BenchmarkStrategy::run_loop, this);
}

void BenchmarkStrategy::run_loop() {
    CUDA_CHECK(cudaSetDevice(gpu_props_.device_id));

    const double test_duration_seconds = 10.0;

    for (const auto& test_desc : tests_to_run_) {
        if (stop_flag_) break;

        BenchmarkResult current_result;
        current_result.descriptor = test_desc;

        // mx-burner 改动：逐项隔离。某些后端对特定 compute/数据类型组合返回
        // CUBLAS_STATUS_NOT_SUPPORTED（例如 mcblasLt 的 INT8 稀疏路径抛异常），
        // 原实现无 try/catch，会让整个进程 terminate，丢掉后面所有项。
        // mx-burner 改动：整项包在 try/catch 内。某些后端对特定 compute/数据类型组合
        // 返回 CUBLAS_STATUS_NOT_SUPPORTED，且是在 execute() 阶段抛出（例如 mcblasLt 的
        // INT8 稀疏路径）。原实现无捕获，会让整个进程 terminate，丢掉后续所有项。
        std::unique_ptr<BaseOperator> op_holder;
        try {
            op_holder = OperatorFactory::create_gemm_operator(test_desc, gpu_props_, current_result.notes);
            BaseOperator* op = op_holder.get();
            if (op == nullptr) {
                if (current_result.notes.empty()) current_result.is_supported = false;
            } else {
                current_result.is_supported = true;
                for (int i = 0; i < 10; ++i) {
                    op->execute(0);
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                // mx-burner 改动：原实现在 10 秒内异步提交成千上万次而不 sync，
                // 在快卡上会积压到分钟级（C550: 4.7us 提交 vs 0.51ms 执行 -> 约 18 分钟）。
                // 改为每 kSyncEvery 次提交同步一次，队列深度有界。
                const int kSyncEvery = 16;
                long long iterations = 0;
                auto start_time = std::chrono::high_resolution_clock::now();
                double elapsed_seconds = 0;

                while (elapsed_seconds < test_duration_seconds) {
                    for (int k = 0; k < kSyncEvery; ++k) {
                        op->execute(0);
                        iterations++;
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());
                    auto end_time = std::chrono::high_resolution_clock::now();
                    elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

                    // Check for throttling during the test
                    if (monitor_.get_state().throttled) {
                        current_result.was_throttled = true;
                    }
                }

                CUDA_CHECK(cudaDeviceSynchronize());
                auto final_end_time = std::chrono::high_resolution_clock::now();
                double precise_duration = std::chrono::duration<double>(final_end_time - start_time).count();

                double total_gops = op->get_gflops_or_gops() * iterations;
                current_result.performance = total_gops / precise_duration / 1000.0;
                current_result.is_native = op->is_native();
                current_result.unit = (test_desc.precision == Precision::INT8) ? "TOPS" : "TFLOPS";
            }
        } catch (const std::exception& e) {
            current_result.is_supported = false;
            current_result.is_native = false;
            current_result.performance = 0.0;
            current_result.notes = std::string("unsupported on this backend: ") + e.what();
            cudaGetLastError();  // 清掉可能残留的错误状态，避免污染后续项
        }
        
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_.push_back(current_result);
        }

        // Pause for 5 seconds between tests to allow GPU to cool down
        if (!stop_flag_) {
            for (int i = 5; i > 0; --i) {
                countdown_seconds_ = i;
                if(stop_flag_) break;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            countdown_seconds_ = 0;
        }
    }
    done_flag_ = true;
}
