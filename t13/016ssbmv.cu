#include <stdio.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define n 6
#define k 1
int main(void){
  cublasHandle_t handle;
  int i,j;
  float* a;
  float* x;
  float* y;
  cudaMallocManaged(&a,n*n*sizeof(float));
  cudaMallocManaged(&x,n*sizeof(float));
  cudaMallocManaged(&y,n*sizeof(float));
  int ind=11;
  for(i = 0; i < n; i++) a[i*n] = (float)ind++;
  for(i = 0; i < n-1; i++) a[i*n+1] = (float)ind++;
  for(i = 0; i < n; i++){x[i]=1.0f; y[i]=0.0f;};

  cublasCreate(&handle);
  float al = 1.0f;
  float bet = 1.0f;
  cublasSsbmv(handle, CUBLAS_FILL_MODE_LOWER,n,k,&al,a,n,x,1,&bet,y,1);
  cudaDeviceSynchronize();
  printf("y after ssbmv:\n");
  for(j = 0; j < n; j++){
    printf("%7.0f\n",y[j]);
  }
  
  cudaFree(a);
  cudaFree(x);
  cudaFree(y);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
