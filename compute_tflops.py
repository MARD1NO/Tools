from loguru import logger 



M = 256
TP = 8 
q_num_head = 64 
k_num_head = 8 
v_num_head = 8
head_size = 128 


HIDDEN_SIZE = 8192
INTERMEDIATE_SIZE = 29568

QKV_PROJ_US = 46
O_PROJ_US = 50
FFN1_PROJ_US = 312
FFN2_PROJ_US = 156


def compute_tflops(M, N, K, time_us): 

    return 2 * M * N * K / time_us / 1e6


QKV_PROJ_K = HIDDEN_SIZE
QKV_PROJ_N = ((q_num_head + k_num_head + v_num_head) * head_size) // TP 

O_PROJ_K = (q_num_head * head_size) // TP 
O_PROJ_N = HIDDEN_SIZE

FFN_1_K = HIDDEN_SIZE
FFN_1_N = (INTERMEDIATE_SIZE) // TP

FFN_2_K = (INTERMEDIATE_SIZE // TP) // 2
FFN_2_N = HIDDEN_SIZE

qkv_tflops = compute_tflops(M, QKV_PROJ_N, QKV_PROJ_K, QKV_PROJ_US)
o_tflops = compute_tflops(M, O_PROJ_N, O_PROJ_K, O_PROJ_US)
ffn_1_tflops = compute_tflops(M, FFN_1_N, FFN_1_K, FFN1_PROJ_US)
ffn_2_tflops = compute_tflops(M, FFN_2_N, FFN_2_K, FFN2_PROJ_US)

logger.info(f"qkv_tflops: {qkv_tflops}")
logger.info(f"o_tflops: {o_tflops}")
logger.info(f"ffn_1_tflops: {ffn_1_tflops}")
logger.info(f"ffn_2_tflops: {ffn_2_tflops}")

