#include <stdio.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define m 5
#define n 6
#define ku 2
#define kl 1
int main(void){
  cublasHandle_t handle;
  int i,j;
  float* a;
  float* x;
  float* y;
  cudaMallocManaged(&a,m*n*sizeof(float));
  cudaMallocManaged(&x,n*sizeof(float));
  cudaMallocManaged(&y,n*sizeof(float));
  int ind=11;
  for(i = ku; i < n; i++) a[IDX2C(0,i,m)]=(float)ind++;
  for(i = ku-1; i < n; i++) a[IDX2C(1,i,m)]=(float)ind++;
  for(i = 0; i < n-1; i++) a[IDX2C(ku,i,m)]=(float)ind++;
  for(i = 0; i < n-2; i++) a[IDX2C(ku+1,i,m)]=(float)ind++;
  
  for(i = 0; i < n; i++) x[i] = 1.0f;
  for(i = 0; i < m; i++) y[i] = 0.0f;
  
  cublasCreate(&handle);
  float al = 1.0f;
  float bet = 1.0f;
  cublasSgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, &al, a, m, x, 1, &bet, y, 1);
  cudaDeviceSynchronize();
  
  printf("y after Sgbmv:\n");
  for(j = 0; j < m; j++){
      printf("%7.0f", y[j]);
      printf("\n");
  }
  cudaFree(a);
  cudaFree(x);
  cudaFree(y);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
