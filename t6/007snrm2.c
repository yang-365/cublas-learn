#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define n 6
int main(void){
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int j;
  float* x;
  x = (float*)malloc(n*sizeof(float));
  for(j = 0; j < n; j++){
    x[j] = (float)j;
  }

  // device
  float* d_x;
  float result;
  cublasCreate(&handle);
  stat = cudaMalloc((void**)&d_x, n*sizeof(float));
  stat = cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
  
  stat = cublasSnrm2(handle, n, d_x, 1, &result);
  
  printf("Euclidean norm: %7.3f\n", result);
  cudaFree(x);
  free(x);
  return EXIT_SUCCESS;
}
