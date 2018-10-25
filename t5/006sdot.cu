#include <stdio.h>
#include "cublas_v2.h"
#define n 6
int main(void){
  cublasHandle_t handle;
  int j;
  float* x;
  float* y;
  cudaMallocManaged(&x, n*sizeof(float));
  cudaMallocManaged(&y, n*sizeof(float));
  for(j = 0; j < n; j++){
    x[j] = (float)j;
    y[j] = (float)j;
  }
  
  // device
  float result;
  cublasCreate(&handle);
  cublasSdot(handle, n, x, 1, y, 1, &result);
  cudaDeviceSynchronize();
  printf("dot product:");
  printf("%7.0f\n", result);
  cudaFree(x);
  cudaFree(y);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
