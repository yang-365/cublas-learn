#include <stdio.h>
#include "cublas_v2.h"
#define n 6
int main(void){
  cublasHandle_t handle;
  int j;
  float* x;
  cudaMallocManaged((void**)&x, n*sizeof(float));
  for(j  = 0; j < n; j++){
    x[j] = (float)j;
  }

  // device
  float result;
  cublasCreate(&handle);
  cublasSnrm2(handle, n, x, 1, &result);
  cudaDeviceSynchronize();
  printf("Euclidean norm: %7.3f\n", result);
  cudaFree(x);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
