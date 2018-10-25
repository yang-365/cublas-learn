#include <stdio.h>
#include "cublas_v2.h"
#define n 6
int main(void){
  cublasHandle_t handle;  // CUBLAS context
  int j;
  float* x;
  float* y;
  cudaMallocManaged(&x, n*sizeof(float));
  for(j  = 0; j < n; j++){
    x[j] = (float)j;
  }
  cudaMallocManaged(&y, n*sizeof(float));
  for(j = 0; j < n; j++){
    y[j] = (float)j;
  }
  printf("x:");
  for(j = 0; j < n; j++){
    printf("%4.0f ", x[j]);
  }
  printf("\n");
  printf("y:");
  for(j = 0; j < n; j++){
    printf("%4.0f ", y[j]);
  }
  printf("\n");

  // device
  cublasCreate(&handle);
  float al = 2.0;
  cublasSaxpy(handle, n, &al, x, 1, y, 1);
  cudaDeviceSynchronize();

  printf("y after Saxpy:");
  for(j = 0; j < n; j++){
    printf("%4.0f ", y[j]);
  }
  printf("\n");

  cudaFree(x);
  cudaFree(y);
  return EXIT_SUCCESS;
}
