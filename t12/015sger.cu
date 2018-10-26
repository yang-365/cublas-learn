#include <stdio.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define m 6
#define n 5
int main(void){
  cublasHandle_t handle;
  int i,j;
  float* a;
  float* x;
  float* y;
  cudaMallocManaged(&a,m*n*sizeof(float));
  cudaMallocManaged(&x,m*sizeof(float));
  cudaMallocManaged(&y,n*sizeof(float));
  
  int ind = 11;
  float al = 2.0f;
  for(j = 0; j < n; j++){ 
    for(i = 0; i < m; i++){
      a[IDX2C(i,j,m)] = (float)ind++;
    }
  }
  for(i = 0; i < m; i++) x[i] = 1.0f;
  for(i = 0; i < n; i++) y[i] = 1.0f;

  cublasCreate(&handle);
  cublasSger(handle,m,n,&al,x,1,y,1,a,m);
  cudaDeviceSynchronize();
  
  printf("a after sger:\n");
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
      printf("%7.0f", a[IDX2C(i,j,m)]);
    }
    printf("\n");
  }
  cudaFree(a);
  cudaFree(x);
  cudaFree(y);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
