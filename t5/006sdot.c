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
  float* y;
  x = (float*)malloc(n*sizeof(float));
  y = (float*)malloc(n*sizeof(float));
  for(j = 0; j < n; j++){
    x[j] = (float)j;
    y[j] = (float)j;
  }

  // on the device
  float* d_x;
  float* d_y;
  stat = cublasCreate(&handle);
  cudaStat = cudaMalloc((void**)&d_x, n*sizeof(float));
  cudaStat = cudaMalloc((void**)&d_y, n*sizeof(float));
  
  stat = cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
  stat = cublasSetVector(n, sizeof(float), y, 1, d_y, 1);
  
  float result;
  stat = cublasSdot(handle, n, d_x, 1, d_y, 1, &result);

  printf("dot product :");
  printf("%7.0f\n", result);
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
