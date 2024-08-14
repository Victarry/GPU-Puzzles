import numba
from numba import cuda
import numpy as np

TPB = 3

@cuda.jit()
def mm_oneblock_test(out, a, b, size: int) -> None:
    a_shared = cuda.shared.array((TPB, TPB), numba.int32)
    b_shared = cuda.shared.array((TPB, TPB), numba.int32)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    
    out_shared = cuda.shared.array((TPB, TPB), numba.int32)
    num_tiles = (size + (TPB-1)) // TPB
    for k in range(num_tiles):
        if k*TPB+local_j < size and i < size:
            a_shared[local_i, local_j] = a[i, k*TPB+local_j]
        if k*TPB+local_i < size and j < size:
            b_shared[local_i, local_j] = b[k*TPB+local_i, j]
        cuda.syncthreads()

        for x in range(TPB):
            if k*TPB + x < size and i < size and j < size:
                out_shared[local_i, local_j] += a_shared[local_i, x] * b_shared[x, local_j]
        cuda.syncthreads()
        out[i, j] = out_shared[local_i, local_j]

# Example usage
size = 8
# A = np.random.randint(0, 100, (size, size)).astype(np.int32) 
# B = np.random.randint(0, 100, (size, size)).astype(np.int32)
A = np.ones((size, size), dtype=np.int32)
B = np.ones((size, size), dtype=np.int32)
C = np.zeros((size, size), dtype=np.int32)

size = A.shape[0]
threads_per_block = (TPB, TPB)
blocks_per_grid_x = (size + TPB - 1) // TPB
blocks_per_grid_y = (size + TPB - 1) // TPB
mm_oneblock_test[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](C, A, B, size)
# print(C)
print(A@B)

print(C)
