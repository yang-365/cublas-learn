#include <stdio.h>
#include "cublas_v2.h"
#define n 6
int main(void){
  cublasHandle_t handle;
  int j;
  float *x;
  cudaMallocManaged(&x, n*sizeof(float));
  for(j = 0; j < n; j++){
    x[j] = (float)j;
  }
  printf("x: ");
  for(j = 0; j < n; j++){
    printf("%4.0f ", x[j]);
  }
  printf("\n");
  // DEVICE
  cublasCreate(&handle);  // initialize CUBLAS context
  float result;
  // add sbsolute value 
  cublasSasum(handle, n, x, 1, &result);
  cudaDeviceSynchronize();
  printf("sum of absolute value %4.0f\n", result);
  
  cudaFree(x);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
  
}
