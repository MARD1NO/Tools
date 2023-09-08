constexpr int LAUNCH_CHECK_NAN_BLOCK_SIZE = 1024; 
constexpr int LAUNCH_CHECK_NAN_GRID_SIZE = 12; 

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);

  return val;
}

__global__ void reset_nan_inf_ptr(int32_t* block_num_nan_ptr,
                                  int32_t* block_num_inf_ptr, 
                                  const int32_t elem_cnt){
  for(int i = 0; i < elem_cnt; i++){
    block_num_nan_ptr[i] = 0; 
    block_num_inf_ptr[i] = 0; 
  }
}

template <typename T, typename MT>
__global__ void FindNanInfAndBlockMaxMin(T* value_ptr,
                                         const int64_t numel,
                                         int32_t* block_num_nan_ptr,
                                         int32_t* block_num_inf_ptr) {
  int64_t i = threadIdx.x + blockIdx.x * blockDim.x;

  int32_t num_nan = 0;
  int32_t num_inf = 0;

  for (; i < numel; i += blockDim.x * gridDim.x) {
    MT value = static_cast<MT>(value_ptr[i]);

    if (isnan(value)) {
      num_nan += 1;
    } else if (isinf(value)) {
      num_inf += 1;
    }
  }
  int32_t block_reduce_sum_nan = blockReduceSum(num_nan); 
  int32_t block_reduce_sum_inf = blockReduceSum(num_inf); 
  if(threadIdx.x == 0){
    atomicAdd(block_num_nan_ptr, block_reduce_sum_nan); 
    atomicAdd(block_num_inf_ptr, block_reduce_sum_inf); 
  }
}