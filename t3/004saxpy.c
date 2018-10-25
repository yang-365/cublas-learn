#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define n 6
int main(void){
  cudaError_t cudaStat;  // cudaMalloc status
  cublasStatus_t stat;  // CUBLAS functions status
  cublasHandle_t handle;  // CUBLAS context
  int j;  // index of elements
  float *x;  // n-vector on the host
  float *y;  // n-vector on the host
  x = (float*)malloc(n*sizeof(*x));
  for(j = 0; j < n; j++){
    x[j] = (float)j;
  }
  printf("x:");
  for(j = 0; j < n; j++){
    printf("%4.0f ", x[j]);
  }
  printf("\n");
  
  y = (float*)malloc(n*sizeof(*x));
  for(j = 0; j < n; j++){
    y[j] = (float)j;
  }
  printf("y:");
  for(j = 0; j < n; j++){
    printf("%4.0f ", y[j]);
  }
  printf("\n");

  // device
  float *d_x;
  float *d_y;
  cudaStat = cudaMalloc((void**)&d_x, n*sizeof(*x));
  cudaStat = cudaMalloc((void**)&d_y, n*sizeof(*y));

  stat = cublasCreate(&handle);
  stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
  stat = cublasSetVector(n, sizeof(*y), y, 1, d_y, 1);

  float al = 2.0;
  // multi d_x by al and add to d_y
  stat = cublasSaxpy(handle, n, &al, d_x, 1, d_y, 1);
  stat = cublasGetVector(n, sizeof(float), d_y, 1, y ,1);
  printf("y after Saxpy : ");
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
