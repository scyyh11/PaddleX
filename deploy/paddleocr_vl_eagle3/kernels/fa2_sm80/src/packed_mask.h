// Copyright (c) 2026. Experimental adaptation of FA2's BSD-licensed mask.h.
#pragma once
#include <cute/tensor.hpp>
#include "namespace_config.h"
namespace FLASH_NAMESPACE {
using namespace cute;
struct PackedRowMask {
    const int* endpoints;
    const int queries;
    __forceinline__ __device__ PackedRowMask(const int* lengths, int count)
        : endpoints(lengths), queries(count) {}
    template<bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine,Layout>& tensor_,
            int col_idx_offset_, int row_idx_offset, int warp_row_stride) {
        static_assert(!Causal_mask, "Packed Q1 rows retain the original noncausal kernel specialization");
        static_assert(Layout::rank == 3);
        Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
        #pragma unroll
        for (int mi = 0; mi < size<0,1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0,0>(tensor); ++i) {
                const int row = row_idx_base + i * 8;
                // M64 includes unused rows for Q5 (40) and Q7 (56).
                const int query = row / 8;
                const int endpoint = query < queries ? endpoints[query] : 0;
                #pragma unroll
                for (int nj = 0; nj < size<1,1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1,0>(tensor); ++j) {
                        if (col_idx_base + j >= endpoint)
                            tensor(make_coord(i,mi),make_coord(j,nj)) = -INFINITY;
                    }
                }
            }
        }
    }
};
}
