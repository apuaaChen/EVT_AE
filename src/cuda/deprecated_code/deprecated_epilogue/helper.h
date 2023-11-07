#ifndef EPILOGUE_HELPER_H
#define EPILOGUE_HELPER_H
#include <stdio.h>

__device__ void print_val(int blockidx, int blockidy, int blockidz, int threadid, float value){
    if (blockidx == blockIdx.x && blockidy == blockIdx.y && blockidz == blockIdx.z && threadid == threadIdx.x) printf("tid: %d, value is: %.8f\n", threadid, float(value));
}

#endif