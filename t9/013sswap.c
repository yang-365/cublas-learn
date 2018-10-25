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
    y[j] = (float)j*j;
  }

  // device
  float* d_x;
  float* d_y;
  cudaStat = cudaMalloc((void**)&d_x, n*sizeof(float));
  cudaStat = cudaMalloc((void**)&d_y, n*sizeof(float));
  
  stat = cublasCreate(&handle);
  stat = cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
  stat = cublasSetVector(n, sizeof(float), y, 1, d_y, 1);
  stat = cublasSswap(handle, n, d_x, 1, d_y, 1);
  stat = cublasGetVector(n, sizeof(float), d_x, 1, x, 1);
  stat = cublasGetVector(n, sizeof(float), d_y, 1, y, 1);
  printf("x after sswap: ");
  for(j = 0; j < n; j++){
    printf("%7.3f ", x[j]);
  }
  printf("\n");
  printf("y after sswap: ");
  for(j = 0; j < n; j++){
    printf("%7.3f ", y[j]);
  }
  printf("\n");
  
  cudaFree(d_x);
  cudaFree(d_y);
  cublasDestroy(handle);
  free(x);
  free(y);
  return EXIT_SUCCESS;
  
}
