#ifndef EPILOGUE_HELPER_H
#define EPILOGUE_HELPER_H
#include <stdio.h>

__device__ void print_val(int blockid, int threadid, float value){
    if (blockid == 0 && threadid == 0) printf("tid: %d, value is: %.8f\n", threadid, float(value));
}

#endif