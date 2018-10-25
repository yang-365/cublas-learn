#include <stdio.h>
#include "cublas_v2.h"
#define n 6
int main(void){
  cublasHandle_t handle;  // CUBLAS context
  int j;  // index of the elements
  float *x;
  cudaMallocManaged(&x, n*sizeof(float));  // unified mem for x
  for(j = 0; j < n; j++){
    x[j] = (float)j;  // x = {0, 1, 2, 3, 4, 5}
  }
  printf("x: ");
  for(j = 0; j < n; j++){
    printf("%4.0f ", x[j]);
  }
  printf("\n");
  // device 
  cublasCreate(&handle);
  int result;
  // find smallset absolute value index
  cublasIsamin(handle, n, x, 1, &result);
  cudaDeviceSynchronize();
  printf("mim |x[i]|:%4.0f\n", fabs(x[result-1]));
  // find maximal absolute value index
  cublasIsamax(handle, n, x, 1, &result);
  printf("max |x[i]|:%4.0f\n", fabs(x[result-1]));
  cudaDeviceSynchronize();
  
  cudaFree(x);
  cublasDestroy(handle);
  return EXIT_SUCCESS;

}
