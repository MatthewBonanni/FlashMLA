#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/barrier.h>

#include "config.h"

using TMABarrier = cutlass::arch::ClusterTransactionBarrier;
using namespace cute;

#include "fp8_transpose_v.h"

template<typename T, int DIM, int DIM2, cute::GMMA::Major major>
constexpr auto getSmemLayout() {
    constexpr int headSizeBytes = sizeof(T) * DIM;
    constexpr int headSizeBytes2 = sizeof(T) * DIM2;

    if constexpr (major == GMMA::Major::K) {
        if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
            return GMMA::Layout_K_SW128_Atom<T>{};
        } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
            return GMMA::Layout_K_SW64_Atom<T>{};
        } else {
            return GMMA::Layout_K_SW32_Atom<T>{};
        }
    } else {
        if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
            return GMMA::Layout_MN_SW128_Atom<T>{};
        } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
            return GMMA::Layout_MN_SW64_Atom<T>{};
        } else {
            return GMMA::Layout_MN_SW32_Atom<T>{};
        }
    }
}

template<typename InputT_, typename OutputT_ = InputT_>
struct Traits {
    using InputT = InputT_;
    using OutputT = OutputT_;
    
    static constexpr int BLOCK_SIZE_M = Config::BLOCK_SIZE_M;
    static constexpr int PAGE_BLOCK_SIZE = Config::PAGE_BLOCK_SIZE;
    static constexpr int HEAD_DIM_K = Config::HEAD_DIM_K;
    static constexpr int HEAD_DIM_V = Config::HEAD_DIM_V;

    static constexpr int NUM_THREADS = 256;

    static_assert(std::is_same_v<InputT, cutlass::bfloat16_t> ||
                  std::is_same_v<InputT, cutlass::half_t> ||
                  std::is_same_v<InputT, cutlass::float_e4m3_t>);
    static_assert(std::is_same_v<OutputT, cutlass::bfloat16_t> ||
                  std::is_same_v<OutputT, cutlass::half_t>);

    static constexpr bool INPUT_IS_FP8 = cute::is_same_v<InputT, cutlass::float_e4m3_t>;

    static constexpr cute::GMMA::Major MmaMajorV = INPUT_IS_FP8 ? GMMA::Major::K : GMMA::Major::MN;

    using TiledMMA_QK_sQ = decltype(make_tiled_mma(
        GMMA::ss_op_selector<InputT, InputT, float, Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>, Int<HEAD_DIM_K>>, GMMA::Major::K, GMMA::Major::K>(),
        Layout<Shape<_1, _1, _1>>{}
    ));

    using TiledMMA_QK_rQ = decltype(make_tiled_mma(
        GMMA::rs_op_selector<InputT, InputT, float, Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>, Int<HEAD_DIM_K>>, GMMA::Major::K, GMMA::Major::K>(),
        Layout<Shape<_1, _1, _1>>{}
    ));

    using TiledMMA_PV_LocalP = decltype(make_tiled_mma(
        GMMA::rs_op_selector<InputT, InputT, float, Shape<Int<BLOCK_SIZE_M>, Int<HEAD_DIM_V/2>, Int<PAGE_BLOCK_SIZE>>, GMMA::Major::K, MmaMajorV>(),
        Layout<Shape<_1, _1, _1>>{}
    ));

    using TiledMMA_PV_RemoteP = decltype(make_tiled_mma(
        GMMA::ss_op_selector<InputT, InputT, float, Shape<Int<BLOCK_SIZE_M>, Int<HEAD_DIM_V/2>, Int<PAGE_BLOCK_SIZE>>, GMMA::Major::K, MmaMajorV>(),
        Layout<Shape<_1, _1, _1>>{}
    ));

    using SmemLayoutQ = decltype(tile_to_shape(
        getSmemLayout<InputT, HEAD_DIM_K, HEAD_DIM_K, GMMA::Major::K>(),
        Shape<Int<BLOCK_SIZE_M>, Int<HEAD_DIM_K>>{}
    ));

    using SmemLayoutK = decltype(tile_to_shape(
        getSmemLayout<InputT, HEAD_DIM_K, HEAD_DIM_V/2, GMMA::Major::K>(),
        Shape<Int<PAGE_BLOCK_SIZE>, Int<HEAD_DIM_K>>{}
    ));

    using SmemLayoutV = decltype(tile_to_shape(
        getSmemLayout<InputT, HEAD_DIM_K, HEAD_DIM_V/2, MmaMajorV>(),
        Shape<Int<PAGE_BLOCK_SIZE>, Int<HEAD_DIM_V/2>>{}
    ));

    using SmemLayoutVT = decltype(composition(
        SmemLayoutV{},
        make_layout(Shape<Int<HEAD_DIM_V>, Int<PAGE_BLOCK_SIZE>>{}, GenRowMajor{})
    ));

    using SmemLayoutP0 = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<InputT>{},
        Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>>{}
    ));

    using rP0Layout = decltype(layout(partition_fragment_C(
        TiledMMA_QK_sQ{},
        Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>>{}
    )));

    using SmemFP8Transpose = SmemTransposeFP8_64x64<PAGE_BLOCK_SIZE, HEAD_DIM_V/2, SmemLayoutK>;
    using SmemLayoutVTMMA = typename SmemFP8Transpose::SmemLayoutVT;

    using SmemVT = cute::conditional_t<
        INPUT_IS_FP8,
        cute::array_aligned<InputT, cosize_v<SmemLayoutVTMMA>>,
        cute::array_aligned<InputT, 0>>;

    struct SharedMemoryPlan {
        cute::array_aligned<InputT, cosize_v<SmemLayoutQ>> smem_sQ;
        cute::array_aligned<InputT, cosize_v<SmemLayoutK>> smem_sK0;
        cute::array_aligned<InputT, cosize_v<SmemLayoutK>> smem_sK1;
        SmemVT smem_sVT0;
        SmemVT smem_sVT1;
        cute::array_aligned<InputT, cosize_v<SmemLayoutP0>> smem_sP0;
        cute::array_aligned<float, BLOCK_SIZE_M> smem_sM;
        cute::array_aligned<float, 2*BLOCK_SIZE_M> sL_reduction_wksp;
        cute::array_aligned<float, BLOCK_SIZE_M> smem_sScale0;
        cute::array_aligned<float, BLOCK_SIZE_M> smem_sScale1;
        TMABarrier barriers_K0[HEAD_DIM_K/64];
        TMABarrier barriers_K1[HEAD_DIM_K/64];
        TMABarrier barrier_Q;
    };
};

template<
    typename ShapeQ, typename TMA_Q,
    typename ShapeK, typename TMA_K,
    typename ShapeO, typename TMA_O
>
struct TmaParams {
    ShapeQ shape_Q;
    TMA_Q tma_Q;
    ShapeK shape_K;
    TMA_K tma_K;
    ShapeO shape_O;
    TMA_O tma_O;
};

enum NamedBarriers : int {
    sScale0Ready = 0,
    sScale1Ready = 1,
    sP0Ready = 2,
    rO1sP0sV0RIssued = 3
};
