from numba import cuda
import numpy as np
import numba

def conv1d_spec(a, b):
    result = np.zeros_like(a)
    m = a.shape[0]
    n = b.shape[0]
    for i in range(m):
        for k in range(n):
            if i + k < m:
                result[i] += a[i+k]*b[k]
    return result


TPB=8
MAX_CONV=4
TPB_MAX_CONV=TPB+MAX_CONV

@cuda.jit
def conv1d(out, a, b) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x

    # FILL ME IN (roughly 17 lines)
    shared_a = cuda.shared.array(TPB_MAX_CONV, dtype=numba.int32)
    shared_b = cuda.shared.array(TPB, numba.int32)

    if i < a.size:
        if local_i < TPB:
            shared_a[local_i] = a[i]
        if local_i < b.size:
            shared_b[local_i] = b[local_i]
            if i + TPB < a.size:
                shared_a[local_i+TPB] = a[i + TPB]
        cuda.syncthreads()

        result = 0
        for k in range(b.size):
            if local_i + k < a.size:
                result += shared_a[local_i+k] * b[k]
        out[i] = result


a = np.arange(20, dtype=np.float32)
b = np.arange(3, dtype=np.float32)
out = np.zeros_like(a)
result = conv1d[2, TPB](out, a, b)

print(a)
print(b)
print(out)
print(conv1d_spec(a, b))