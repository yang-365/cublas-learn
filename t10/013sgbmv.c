#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define m 5  // number of rows
#define n 6  // number of colums
#define ku 2  // number of superdiagonals
#define kl 1  // number of subdiagonals
int main(void){
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int i, j;
  float* a;
  float* x;
  float* y;
  a = (float*)malloc(m*n*sizeof(float));  // m*n matrix on the host
  x = (float*)malloc(n*sizeof(float));  // n-vector on the host
  y = (float*)malloc(m*sizeof(float));  // m-vector on the host
  int ind = 11;
  for(i = ku; i < n; i++){
    a[IDX2C(0,i,m)]=(float)ind++;
  }
  for(i = ku-1; i < n; i++){
    a[IDX2C(1,i,m)]=(float)ind++;
  }
  for(i = 0; i < n-1; i++){
    a[IDX2C(ku,i,m)]=(float)ind++;
  }
  for(i = 0; i < n-2; i++){
    a[IDX2C(ku+1,i,m)]=(float)ind++;
  }
  for(i = 0; i < n; i++){
    x[i] = 1.0f;
  } 
  for(i = 0; i < m; i++){
    y[j] = 0.0f;
  }
  
  // device
  float* d_a;
  float* d_x;
  float* d_y;
  cudaStat = cudaMalloc((void**)&d_a, m*n*sizeof(float));
  cudaStat = cudaMalloc((void**)&d_x, n*sizeof(float));
  cudaStat = cudaMalloc((void**)&d_y, m*sizeof(float));

  stat = cublasCreate(&handle);
  stat = cublasSetMatrix(m,n,sizeof(float),a,m,d_a,m);
  stat = cublasSetVector(n,sizeof(float),x,1,d_x,1);
  stat = cublasSetVector(m,sizeof(float),y,1,d_y,1);
  float al=1.0f;
  float bet=1.0f;
  // d_y = al*d_a*d_x + bet*d_y;  d_a - m*n banded matrix
  // d_x - n-vector  d_y - m-vector;  al,bet - scalars
  stat = cublasSgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, &al, d_a, m, d_x, 1, &bet, d_y, 1);
  stat = cublasGetVector(m,sizeof(float),d_y,1,y,1);
  printf("y after Sgbmv:\n");
  for(j = 0; j < m; j++){
    printf("%7.0f\n", y[j]);
    printf("\n");
  }
  cudaFree(d_a);
  cudaFree(d_x);
  cudaFree(d_y);
  cublasDestroy(handle);
  free(a);
  free(x);
  free(y);
  return EXIT_SUCCESS;
}  
