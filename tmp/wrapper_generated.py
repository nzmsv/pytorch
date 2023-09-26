
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_klondenberg/pu/cpul3klbbii7ehtnxp2gyskolqqhafljq4yidg3tgvcbkdfedsdy.py
# Source Nodes: [mul], Original ATen: [aten.mul]
# mul => mul
cuda_fused_mul_0 = async_compile.cuda(r'''
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/distribution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"


// We compile all models with -fvisibility=hidden. Any symbols that need to be
// exposed in the final shared library must be declared with PT_EXPORT to make
// them visible.
#ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
#define PT_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define PT_EXPORT __declspec(dllexport)
#else
#define PT_EXPORT
#endif
#endif
using bfloat16 = nv_bfloat16;

using namespace cute;
#define CUTLASS_CHECK(status)                                                      \
{                                                                                  \
  cutlass::Status error = status;                                                  \
  if (error != cutlass::Status::kSuccess) {                                        \
    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \
        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \
    throw std::runtime_error(msg);                                                 \
  }                                                                                \
}


using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
static_assert(cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecialized> ||
         cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecializedCooperative>,
        "Epilogue visitor trees are currently only supported by the TMA warp-specialized epilogue");
static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
using ElementAcc = cutlass::half_t;
using EVT_expr_1 = cutlass::epilogue::fusion::Sm90AccFetch /* :=buf1 (matmul output in accumulator) */;
using EVT_expr_2 = cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementAcc> /* value=3.0, dtype=torch.float16 */;
using EVT_expr_3 = cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementAcc, ElementAcc, RoundStyle>,EVT_expr_1,EVT_expr_2>;
using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_epilogue_functor = EVT_expr_3;
;
using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::half_t, cutlass::half_t,
    void, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    EpilogueScheduleType,
    cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_epilogue_functor
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_epilogue::SharedStorage)>,
  cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma
using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_mainloop,
    cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_epilogue,
    cutlass::gemm::PersistentScheduler>;

// Define named type
struct cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma :
  public cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_base { };


  using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_device_type = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma>;

// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, compuates the Gemm kernel using the given workspace ptr.
extern "C" {
PT_EXPORT int cuda_fused_mul_0(const half* X, const half* W, half* Y, size_t* workspace_size, uint8_t* workspace, cudaStream_t stream) {
  try {

  {
    if (!X) {
      int64_t X_size = 8192L;
      if (X_size > 0) {
        throw std::runtime_error("input X is null but size is not 0!");
      }
    }
  }


  {
    if (!W) {
      int64_t W_size = 8192L;
      if (W_size > 0) {
        throw std::runtime_error("input W is null but size is not 0!");
      }
    }
  }



  {
    if (!Y) {
      int64_t Y_size = 65536L;
      if (Y_size > 0) {
        throw std::runtime_error("input Y is null but size is not 0!");
      }
    }
  }

  int64_t B = 1;
  int64_t M = 256L;
  int64_t K = 32L;
  int64_t N = 256L;
  using ElementComputeEpilogue = cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_device_type::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_device_type::Arguments arguments;

  // Initialize GemmUniversal3xInstance arguments.
  arguments = {
    cutlass::gemm::GemmUniversalMode::kGemm,  // GemmUniversalMode mode
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(K),
      static_cast<coord_t>(B)
    }, // ProblemShape problem_shape
    {
      (cutlass::half_t*)(X),  // ElementA const* ptr_A
      {
        32L /* stride_x0 */,
        cute::Int<1>{} /* stride_x1 */,
        0 /* batch_stride_x */
      },  // StrideA dA
      (cutlass::half_t*)(W),  // ElementB const* ptr_B
      {
        cute::Int<1>{} /* stride_w1 */,
        256L /* stride_w0 */,
        0 /* batch_stride_w */
      },  // StrideB dB
    },  // MainloopArguments mainloop

    // see https://github.com/NVIDIA/cutlass/blob/e0aaa3c3b38db9a89c31f04fef91e92123ad5e2e/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp#L184
    {
      {{}, { static_cast<ElementAcc>(3.0) }},  // thread, typename FusionCallbacks::Arguments ( EVT Arguments )
      nullptr,  // ElementC const* ptr_C
      {
        cute::Int<1>{} /* stride_bias0 */,
        cute::Int<1>{} /* stride_bias1 */,
        0 /* batch_stride_bias */
      },  // StrideC dC
      (cutlass::half_t*)(Y),  // ElementD const* ptr_D
      {
        256L /* stride_y0 */,
        cute::Int<1>{} /* stride_y1 */,
        0 /* batch_stride_y */
      },  // StrideD dD
    },  // EpilogueArguments epilogue
  };
  cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_cooperative_epi_tma_device_type gemm_op;
  if (workspace_size) {
    *workspace_size = gemm_op.get_workspace_size(arguments);
    return 0;
  }
  {
    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
  }
  {
    auto status = gemm_op.initialize(arguments, workspace, stream);
    CUTLASS_CHECK(status);
  }
  {
    auto status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
  }
  catch (std::exception& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return -1;
  }
  catch (...) {
    return -1;
  }
  return 0;
}
}
''', 'so')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (256, 32), (32, 1))
    assert_size_stride(arg1_1, (32, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty_strided((256, 256), (256, 1), device='cuda', dtype=torch.float16)
        buf4 = empty_strided((256, 256), (256, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_cuda_stream(0)
        cuda_fused_mul_0.cuda_fused_mul_0(c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf4.data_ptr()), None, None, c_void_p(stream0))
        del arg0_1
        del arg1_1
        return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((32, 256), (256, 1), device='cuda:0', dtype=torch.float16)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
