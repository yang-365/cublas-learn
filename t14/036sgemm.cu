#include <stdio.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define m 6
#define n 4
#define k 5
int main(void){
  cublasHandle_t handle;
  int i,j;
  float* a;
  float* b;
  float* c;
  
  cudaMallocManaged(&a,m*k*sizeof(float));
  cudaMallocManaged(&b,k*n*sizeof(float));
  cudaMallocManaged(&c,m*n*sizeof(float));

  int ind = 11;
  for(j = 0; j < k; j++){
    for(i = 0; i < m; i++){
      a[IDX2C(i,j,m)] = (float)ind++;
    }
  }

  ind = 11;
  for(j = 0; j < n; j++){
    for(i = 0; i < k; i++){
      b[IDX2C(i,j,k)] = (float)ind++;
    }
  }
  
  ind = 11;
  for(j = 0; j < n; j++){
    for(i = 0; i < m; i++){
      c[IDX2C(i,j,m)] = (float)ind++;
    }
  }
  
  cublasCreate(&handle);
  float al = 1.0f;
  float bet = 1.0f;
  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,c,m);
  cudaDeviceSynchronize();
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      printf("%7.0f ",c[IDX2C(i,j,m)]);
    }
    printf("\n");
  }
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
