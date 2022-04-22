#ifndef EPILOGUE_HELPER_H
#define EPILOGUE_HELPER_H
#include <stdio.h>

__device__ void print_val(int blockidx, int blockidy, int threadid, float value){
    if (blockidx == blockIdx.x && blockidy == blockIdx.y && threadid == threadIdx.x) printf("tid: %d, value is: %.8f\n", threadid, float(value));
}

#endif