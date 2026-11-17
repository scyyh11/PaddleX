// Experimental FA2 Q1-preserving row packing; only valid for finite Q/K/V.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "flash.h"
#include "kernel_traits.h"
#include "utils.h"
#include "packed_mask.h"
#include "packed_flash_fwd_kernel_v3.h"

using Traits = Flash_fwd_kernel_traits<128,64,128,4,false,false,cutlass::bfloat16_t>;
struct PackedParams : FLASH_NAMESPACE::Flash_fwd_params {
    const int* packed_endpoints;
    int packed_queries;
};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define PACKED_GRID_CONSTANT __grid_constant__
#else
#define PACKED_GRID_CONSTANT
#endif

__global__ void prepare_packed(const uint16_t* q, uint16_t* packed,
        const int* cu, const int* lengths, int* endpoints, int* maximum,
        int queries, int64_t row_stride, int64_t head_stride) {
    for (int idx=blockIdx.x*blockDim.x+threadIdx.x; idx<queries*16*128; idx+=blockDim.x*gridDim.x) {
        int dim=idx%128;
        int head=(idx/128)%2;
        int row=idx/(128*2);
        int token=row/8, group=row%8;
        packed[idx]=q[token*row_stride+(head*8+group)*head_stride+dim];
    }
    if (blockIdx.x==0 && threadIdx.x < queries) {
        int row=threadIdx.x;
        bool valid=row>=cu[0] && row<cu[1] && lengths[0]>0;
        endpoints[row]=valid ? max(lengths[0]-cu[1]+row+1,0) : 0;
    }
    if (blockIdx.x==0 && threadIdx.x==0) {
        int last=min(queries-1,cu[1]-1);
        bool valid=last>=cu[0] && last>=0 && lengths[0]>0;
        maximum[0]=valid ? max(lengths[0]-cu[1]+last+1,0) : 0;
    }
}

__global__ void packed_main(PACKED_GRID_CONSTANT const PackedParams params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    FLASH_NAMESPACE::compute_attn_1rowblock_splitkv<Traits,
        false,false,false,false,true,false,true,false>(params,0,blockIdx.z,0,blockIdx.y,gridDim.y);
#endif
}

template<int LogSplits>
__global__ void packed_combine(PACKED_GRID_CONSTANT const PackedParams params) {
    FLASH_NAMESPACE::combine_attn_seqk_parallel<Traits,4,LogSplits,true>(params);
}

__global__ void unpack_output(const uint16_t* packed, uint16_t* out,
        int queries, int64_t row_stride, int64_t head_stride) {
    for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<queries*16*128;idx+=blockDim.x*gridDim.x) {
        int dim=idx%128, head=(idx/128)%16, token=idx/(128*16);
        int packed_row=token*8+head%8;
        out[token*row_stride+head*head_stride+dim]=packed[(packed_row*2+head/8)*128+dim];
    }
}

extern "C" int packed_fa2_prepare() {
    static_assert(Traits::kNThreads==128);
    static_assert(Traits::kBlockM==64 && Traits::kBlockN==128);
    return int(cudaFuncSetAttribute(packed_main,cudaFuncAttributeMaxDynamicSharedMemorySize,Traits::kSmemSize));
}
extern "C" int packed_fa2_smem() { return Traits::kSmemSize; }

// strides: q(row,head), k(block,row,head), v(block,row,head), out(row,head).
extern "C" int packed_fa2_launch(
        const void* q,const void* k,const void* v,const int* cu,const int* lengths,const int* table,
        void* packed_q,void* packed_out,void* out,int* endpoints,int* maximum,
        void* lse,void* partial_lse,void* partial_out,const int64_t* strides,
        int queries,int max_k,int page_size,int table_columns,float scale,void* stream_ptr) {
    if(queries<2 || queries>7 || (max_k!=4096 && max_k!=12288) || page_size!=16 || table_columns<max_k/16) return -1;
    cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    int splits=max_k/128;
    PackedParams params={};
    params.q_ptr=packed_q;params.k_ptr=const_cast<void*>(k);params.v_ptr=const_cast<void*>(v);params.o_ptr=packed_out;
    params.q_row_stride=2*128;params.q_head_stride=128;params.q_batch_stride=queries*16*128;
    params.k_batch_stride=strides[2];params.k_row_stride=strides[3];params.k_head_stride=strides[4];
    params.v_batch_stride=strides[5];params.v_row_stride=strides[6];params.v_head_stride=strides[7];
    params.o_row_stride=2*128;params.o_head_stride=128;params.o_batch_stride=queries*16*128;
    params.b=1;params.h=params.h_k=2;params.h_h_k_ratio=1;
    params.seqlen_q=queries*8;params.seqlen_k=max_k;params.total_q=queries*8;
    params.seqlen_q_rounded=128;params.seqlen_k_rounded=max_k;
    params.d=params.d_rounded=128;
    params.scale_softmax=scale;params.scale_softmax_log2=scale*M_LOG2E;
    params.p_dropout=1.f;params.p_dropout_in_uint8_t=255;params.rp_dropout=1.f;params.scale_softmax_rp_dropout=scale;
    params.window_size_left=params.window_size_right=-1;params.softcap=0.f;
    params.seqused_k=maximum;params.block_table=const_cast<int*>(table);
    params.block_table_batch_stride=table_columns;params.page_block_size=page_size;
    params.softmax_lse_ptr=lse;params.softmax_lseaccum_ptr=partial_lse;params.oaccum_ptr=partial_out;
    params.is_bf16=true;params.is_causal=false;params.is_seqlens_k_cumulative=true;
    params.unpadded_lse=true;params.seqlenq_ngroups_swapped=true;params.num_splits=splits;
    params.packed_endpoints=endpoints;params.packed_queries=queries;
    prepare_packed<<<(queries*16*128+255)/256,256,0,stream>>>(reinterpret_cast<const uint16_t*>(q),reinterpret_cast<uint16_t*>(packed_q),cu,lengths,endpoints,maximum,queries,strides[0],strides[1]);
    packed_main<<<dim3(1,splits,2),128,Traits::kSmemSize,stream>>>(params);
    int blocks=(2*queries*8+3)/4;
    if(splits==32) packed_combine<5><<<blocks,128,0,stream>>>(params);
    else packed_combine<7><<<blocks,128,0,stream>>>(params);
    unpack_output<<<(queries*16*128+255)/256,256,0,stream>>>(reinterpret_cast<const uint16_t*>(packed_out),reinterpret_cast<uint16_t*>(out),queries,strides[8],strides[9]);
    return int(cudaPeekAtLastError());
}
