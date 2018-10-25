#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define n 6
int main(void){
  cudaError_t cudaStat;  // cudaMemory status
  cublasStatus_t stat;  // CUBLAS functions status
  cublasHandle_t handle;  // CUBLAS context
  int j;
  float* x;
  float* y;
  x = (float*)malloc(n*sizeof(float));
  for(j = 0; j < n; j++){
    x[j] = (float)j;
  }  
  y = (float*)malloc(n*sizeof(float));
  
  // device 
  float* d_x;
  float* d_y;
  cudaStat = cudaMalloc((void**)&d_x, n*sizeof(float));
  cudaStat = cudaMalloc((void**)&d_y, n*sizeof(float));

  stat = cublasCreate(&handle);
  stat = cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
  stat = cublasScopy(handle, n ,d_x, 1, d_y, 1);
  stat = cublasGetVector(n, sizeof(float), d_x, 1, y, 1);
  printf("y after scopy:");
  for(j = 0; j < n; j++){
    printf("%4.0f ", y[j]);
  }
  printf("\n");

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
  return EXIT_SUCCESS;


}
