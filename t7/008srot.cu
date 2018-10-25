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
    y[j] = (float)j*j;
  }

  // device
  float c = 0.5;
  float s = 0.8669254;
  cublasCreate(&handle);
  cublasSrot(handle, n, x, 1, y, 1, &c, &s);
  cudaDeviceSynchronize();
  printf("x after srot:");
  for(j = 0; j < n; j++){
    printf("%7.2f ", x[j]);
  }
  printf("\n");

  printf("y after srot:");
  for(j = 0; j < n; j++){
    printf("%7.2f ", y[j]);
  }
  printf("\n");
  cudaFree(x);
  cudaFree(y);
  cublasDestroy(handle);
  return EXIT_SUCCESS;

}
