/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_fp16.h>
#include <numeric>
#include <iostream>

#include "camera-bevpool.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

namespace bevfusion {
namespace camera {

#define tile_size 10

// The __align__(4) flag can help the compiler get more efficient instructions.
typedef struct __align__(4){
  half val[tile_size];
} combined_half;

static __global__ void bevpool_half_pack10_kernel(const half* camera_feature, const half* depth_weights, unsigned int nchannel,
                                                  const int3* intervals, unsigned int n_intervals, const unsigned int* indices,
                                                  unsigned int out_h, unsigned int out_w, unsigned int ndepth, unsigned int farea,
                                                  half* output_bevfeat, unsigned int ncamera) {
  int interval_index = blockIdx.y * blockDim.y + threadIdx.y;
  int feature_block = threadIdx.x * tile_size;

  if (interval_index >= n_intervals) return;
  int3 interval = intervals[interval_index];
  float accumulate[tile_size] = {0.0f};

  for (int i = interval.x; i < interval.y; i++) {
    int indice = indices[i];
    int camera_index = indice / (ndepth * farea);
    int fm_inner_index = indice % farea;
    float depth_weight = __half2float(depth_weights[indice]);

    unsigned int camera_feature_offset = (camera_index * farea + fm_inner_index) * nchannel + feature_block;
    if (camera_feature_offset + tile_size >= ncamera * nchannel * ndepth * farea) {
        printf("ERROR: camera_feature_offset=%u exceeds camera_feature memory\n", camera_feature_offset);
        asm("trap;");
    }
    combined_half feature = *(combined_half*)(camera_feature + camera_feature_offset);

#pragma unroll
    for (int j = 0; j < tile_size; j++) {
      // Using fma instead of __hfma can avoids cumulative errors and gives more accurate results.
      accumulate[j] = fma(__half2float(feature.val[j]), depth_weight, accumulate[j]);
    }
  }

#pragma unroll
  for (int j = 0; j < tile_size; j++) {
    unsigned int output_offset = (feature_block + j) * out_h * out_w + interval.z;
    output_bevfeat[output_offset] = __float2half(accumulate[j]);
  }
}

class BEVPoolImplement : public BEVPool {
 public:
  virtual ~BEVPoolImplement() {
    if (output_feature_) checkRuntime(cudaFree(output_feature_));
  }

  bool init(const std::vector<int>& camera_shape, unsigned int bev_width, unsigned int bev_height) {
    this->camera_shape_ = camera_shape;
    this->bev_width_ = bev_width;
    this->bev_height_ = bev_height;

    unsigned int C = camera_shape_[1];
    std::cout << "C: " << C << std::endl;
    std::cout << "bev_width_: " << bev_width << std::endl;
    std::cout << "bev_height_: " << bev_height << std::endl;
    volumn_output_ = C * bev_width * bev_height;
    std::cout << "volumn_output_: " << volumn_output_ << std::endl;
    output_dims_ = {1, (int)C, (int)bev_height, (int)bev_width};
    size_t bytes = volumn_output_ * sizeof(nvtype::half);
    size_t aligned_bytes = (bytes + 255) & ~255; // 256 字节对齐
    checkRuntime(cudaMalloc(&output_feature_, aligned_bytes));
//     checkRuntime(cudaMemset(output_feature_, 0, aligned_bytes)); // 测试
    if (output_feature_ == nullptr) {
        std::cerr << "cudaMalloc failed for output_feature_" << std::endl;
        return false;
    }
    return true;
  }

  virtual std::vector<int> shape() override { return output_dims_; }

  virtual nvtype::half* forward(const nvtype::half* camera_feature, const nvtype::half* depth_weights,
                                const unsigned int* indices, const nvtype::Int3* intervals, unsigned int num_intervals,
                                void* stream = nullptr) override {
    unsigned int N, C, D, H, W;
    N = camera_shape_[0];
    C = camera_shape_[1];
    D = camera_shape_[2];
    H = camera_shape_[3];
    W = camera_shape_[4];

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    int thread_x = C / tile_size;
    int thread_y = 1024 / thread_x;
    dim3 threads(thread_x, thread_y);
    dim3 blocks(1, int((num_intervals + thread_y - 1) / thread_y));
    checkRuntime(cudaMemsetAsync(output_feature_, 0x00, volumn_output_ * sizeof(half), _stream));
    checkRuntime(cudaStreamSynchronize(_stream));
    checkKernel(bevpool_half_pack10_kernel<<<blocks, threads, 0, _stream>>>(
        reinterpret_cast<const half*>(camera_feature), reinterpret_cast<const half*>(depth_weights), C,
        reinterpret_cast<const int3*>(intervals), num_intervals, indices, bev_height_, bev_width_, D, W * H, output_feature_, N));

    return reinterpret_cast<nvtype::half*>(output_feature_);
  }

 private:
  unsigned int bev_width_ = 0;
  unsigned int bev_height_ = 0;
  std::vector<int> camera_shape_;  // N(num camera), C(feature), D(depth), H(height), W(width)  (6, 80, 118, 32, 88)
  half* output_feature_ = nullptr;
  std::vector<int> output_dims_;
  unsigned int volumn_output_ = 0;
};

std::shared_ptr<BEVPool> create_bevpool(const std::vector<int>& camera_shape, unsigned int bev_width, unsigned int bev_height) {
  std::shared_ptr<BEVPoolImplement> instance(new BEVPoolImplement());
  if (!instance->init(camera_shape, bev_width, bev_height)) {
    instance.reset();
  }
  return instance;
}

};  // namespace camera
};  // namespace bevfusion