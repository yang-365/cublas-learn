#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) ((j)*(ld)+(i))
#define m 6
#define n 5
int main(void){
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int i,j;
  float* a;
  float* x; 
  float* y;
  a = (float*)malloc(m*n*sizeof(float));
  x = (float*)malloc(m*sizeof(float));
  y = (float*)malloc(n*sizeof(float));
  
  float ind = 11;
  for(j = 0; j < n; j++){
    for(i = 0; i < m; i++){
      a[IDX2C(i,j,m)] = (float)ind++;
    }
  }
  for(i = 0; i < m; i++) x[i]=1.0f;  // x={1,1,1,1,1,1}^T
  for(i = 0; i < n; i++) y[i]=1.0f;  // y={1,1,1,1,1}^T

  float* d_a;
  float* d_x;
  float* d_y;
  float al = 2.0f;
  cudaStat = cudaMalloc((void**)&d_a, m*n*sizeof(float));
  cudaStat = cudaMalloc((void**)&d_x, m*sizeof(float));
  cudaStat = cudaMalloc((void**)&d_y, n*sizeof(float));
  stat = cublasCreate(&handle);
  stat = cublasSetMatrix(m,n,sizeof(float),a,m,d_a,m);
  stat = cublasSetVector(m,sizeof(float),x,1,d_x,1);
  stat = cublasSetVector(n,sizeof(float),y,1,d_y,1);
  // d_a = al*d_x*d_y^T + d_a
  // 13  19  25  31  37
  // 14  20  26  32  38
  // 15  21  27  33  39
  // 16  22  28  34  40
  // 17  23  29  35  41
  // 18  24  30  36  42
  //
  //    [1]               [11 17 23 29 35]  
  //    [1]               [12 18 24 30 36]
  //    [1]               [13 19 25 31 37]
  //= 2*[ ]*[1,1,1,1,1] + [              ]
  //    [1]               [14 20 26 32 38]
  //    [1]               [15 21 27 33 39]
  //    [1]               [16 22 28 34 40]
  stat = cublasSger(handle,m,n,&al,d_x,1,d_y,1,d_a,m);
  stat = cublasGetMatrix(m,n,sizeof(float),d_a,m,a,m);
  printf("a after Sger:\n");
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      printf("%7.0f ", a[IDX2C(i,j,m)]);
    }
    printf("\n");
  }  
  cudaFree(d_a);
  cudaFree(d_x);
  cudaFree(d_y);
  free(a);
  free(x);
  free(y);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
