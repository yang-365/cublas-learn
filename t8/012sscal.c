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
  float al = 2.0;
  cudaStat = cudaMalloc((void**)&d_x, n*sizeof(float));
  
  stat = cublasCreate(&handle);
  stat = cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
  stat = cublasSscal(handle, n, &al, d_x, 1);
  stat = cublasGetVector(n, sizeof(float), d_x, 1, x, 1);
  printf("x after srot: ");
  for(j = 0; j < n; j++){
    printf("%7.3f ", x[j]);
  }
  printf("\n");
  
  cudaFree(d_x);
  cublasDestroy(handle);
  free(x);
  return EXIT_SUCCESS;
  
}
