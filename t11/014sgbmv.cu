#include <stdio.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))  // 将一行数组转化成一个矩阵 先纵向一列，j为列数，i为行数
#define m 6
#define n 5
int main(void){
  cublasHandle_t handle;
  int i, j;
  float* a;
  float* x;
  float* y;
  cudaMallocManaged(&a,m*n*sizeof(float));
  cudaMallocManaged(&x,n*sizeof(float));
  cudaMallocManaged(&y,n*sizeof(float));
  int ind=11;
  for(j = 0; j < n; j++){
    for(i = 0; i < m; i++){
      a[IDX2C(i,j,m)] = (float)ind++;
    }
  }
  for(i = 0; i < n; i++) x[i]=1.0f;
  for(i = 0; i < m; i++) y[i]=0.0f;

  // device
  cublasCreate(&handle);
  float al = 1.0f;
  float bet = 0.0f;
  cublasSgemv(handle, CUBLAS_OP_N, m, n, &al, a, m, x, 1, &bet, y, 1);
  cudaDeviceSynchronize();
  printf("y after sgemv:\n");
  for(j = 0; j < m; j++){
    printf("%7.0f\n", y[j]);
    printf("\n");
  }
  cudaFree(a);
  cudaFree(x);
  cudaFree(y);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
