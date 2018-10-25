#include <stdio.h>
#include "cublas_v2.h"
#define n 6
int main(void){
  cublasHandle_t handle;
  int j;
  float *x;
  cudaMallocManaged((void**)&x, n*sizeof(float));
  for(j = 0; j < n; j++){
    x[j] = (float)j;
  }

  cublasCreate(&handle);
  float al = 2.0;
  cublasSscal(handle, n, &al, x, 1);
  cudaDeviceSynchronize();
  printf("x after sscal:");
  for(j = 0; j < n; j++){
    printf("%7.3f ", x[j]);
  }
  printf("\n");

  cudaFree(x);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
