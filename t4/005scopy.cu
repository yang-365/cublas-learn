#include <stdio.h>
#include "cublas_v2.h"
#define n 6
int main(void){
  cublasHandle_t handle;  // CUBLAS context
  int j;
  float* x;
  float* y;
  cudaMallocManaged((void**)&x, n*sizeof(float));
  for(j = 0; j < n; j++){
    x[j] = (float)j;
  }  
  cudaMallocManaged((void**)&y, n*sizeof(float));
  
  // device 
  cublasCreate(&handle);
  cublasScopy(handle, n ,x, 1, y, 1);
  cudaDeviceSynchronize();
  printf("y after scopy:");
  for(j = 0; j < n; j++){
    printf("%4.0f ", y[j]);
  }
  printf("\n");

  cudaFree(x);
  cudaFree(y);
  cublasDestroy(handle);
  return EXIT_SUCCESS;

}
