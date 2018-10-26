#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define n 6
#define k 1
int main(void){
  cudaError_t cudaStat; 
  cublasStatus_t stat;
  cublasHandle_t handle;
  int i,j;
  float* a;
  float* x;
  float* y;
  a = (float*)malloc(n*n*sizeof(float));
  x = (float*)malloc(n*sizeof(float));
  y = (float*)malloc(n*sizeof(float));
  
  int ind = 11;
  for(i = 0; i < n; i++) a[i*n] = (float)ind++;  // main diagonal
  for(i = 0; i < n-1; i++) a[i*n+1] = (float)ind++;  // first subdiag
  for(i = 0; i < n; i++) {x[i] = 1.0f; y[i] = 0.0f;};  // x={1,1,1,1,1,1}^T  y={0,0,0,0,0,0}^T

  float* d_a;
  float* d_x;
  float* d_y;
  cudaStat = cudaMalloc((void**)&d_a, n*n*sizeof(float));
  cudaStat = cudaMalloc((void**)&d_x, n*sizeof(float));
  cudaStat = cudaMalloc((void**)&d_y, n*sizeof(float));

  stat = cublasCreate(&handle);
  stat = cublasSetMatrix(n,n,sizeof(float),a,n,d_a,n);
  stat = cublasSetVector(n,sizeof(float),x,1,d_x,1);
  stat = cublasSetVector(n,sizeof(float),y,1,d_y,1);
  float al = 1.0f;
  float bet = 1.0f;
  // d_y = al*d_a*dx + bet*d_y
  // 28     [11 17            ] [1]
  // 47     [17 12 18         ] [1]
  // 50  =  [   18 13 19      ] [1]
  // 53     [      19 14 20   ] [1]
  // 56     [         20 15 21] [1]
  // 37     [            21 16] [1]
  stat = cublasSsbmv(handle,CUBLAS_FILL_MODE_LOWER,n,k,&al,d_a,n,d_x,1,&bet,d_y,1);
  stat = cublasGetVector(n,sizeof(float),d_y,1,y,1);
  printf("y after ssbmv:\n");
  for(j = 0; j < n; j++){
    printf("%7.0f\n", y[j]);
  }
  
  cudaFree(a);
  cudaFree(x);
  cudaFree(y);
  free(a);
  free(x);
  free(y);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
