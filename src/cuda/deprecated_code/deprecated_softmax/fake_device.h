#pragma once

#if defined(__NVCC__)
#else
/// cub header requires a set of device functions that are unavailable in g++
/// We use a set of fake functions to workaround this issue
void __syncthreads(){}
// void cudaGetLastError(){}
int __syncthreads_and(int p){}
int __syncthreads_or(int p){}
int __any(int p){}
int __all(int p){}
int __ballot(int p){}
int __shfl(unsigned int p, int k){}
int __syncwarp(int p){}
int __any_sync(unsigned maks, int predicate){}
int __all_sync(unsigned maks, int predicate){}
int __ballot_sync(unsigned maks, int predicate){}
int __shfl_sync(unsigned mask, float var, int srcLane){}
#endif