from numba import cuda
import numpy as np
import numba

TPB=8
# Define the CUDA kernel using shared memory
@cuda.jit
def prefix_sum(out, a):
    cache = cuda.shared.array(TPB, numba.float32)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x
    # FILL ME IN (roughly 12 lines)
    if i < a.size:
        cache[local_i] = a[i]
        cuda.syncthreads()

        stride = 2 # which range the target prefix sum to be calculated
        while stride <= TPB:
            # Only thread in the right part of range need to accmulate
            if local_i % stride >= stride // 2:
                # Get the prefix sum of the left part of the range
                previous_sum = cache[local_i - local_i % (stride // 2) - 1]
                cache[local_i] = cache[local_i] + previous_sum
            cuda.syncthreads()
            stride *= 2
        # if i == a.size - 1: 
            # out[0] = cache[i % TPB]
        # elif local_i == TPB-1:
            # out[0] = cache[local_i]
        out[i] = cache[local_i]

# Create a numpy array
array = np.arange(10, dtype=np.int32)
out = np.zeros_like(array)

# Copy the array to the device
a_device = cuda.to_device(array)
out_device = cuda.to_device(out)

# Define the number of threads per block and blocks per grid
threads_per_block = TPB
blocks_per_grid = (array.size + (threads_per_block - 1)) // threads_per_block

# Launch the kernel
prefix_sum[blocks_per_grid, threads_per_block](out_device, a_device)

# Copy the result back to the host
result = out_device.copy_to_host()

print('Result:', result)