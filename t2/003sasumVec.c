#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define n 6  // length of x
int main(void){
  cudaError_t cudaStat;  // cudaMalloc status
  cublasStatus_t stat;  // CUBLAS functions status
  cublasHandle_t handle;  // CUBLAS context
  int j;
  float *x;
  x = (float*)malloc(n*sizeof(*x));
  for(j = 0; j < n; j++){
    x[j] = (float)-j;
  }
  printf("x:");
  for(j = 0; j < n; j++){
    printf("%4.0f ", x[j]);
  }
  printf("\n");
  
  // on the device
  float* d_x;
  cudaStat = cudaMalloc((void**)&d_x, n*sizeof(*x));
  stat = cublasCreate(&handle);
  stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1); 
  float result;
  stat = cublasSasum(handle, n, d_x, 1, &result);
  printf("sum of the absolute values of elements of x:%4.0f\n", result);
  cudaFree(d_x);
  cublasDestroy(handle);
  free(x);
  return EXIT_SUCCESS;
}
