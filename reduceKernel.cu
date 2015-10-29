#include "inttypes.h"
#include "cudaErrorHadling.h"
#define BLOCK_SIZE 512

__global__ void initKernel(float *f, int32_t n){
	int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n){
		float h = 1.0f/n;
		float x = (0.5f+i)*h;
		f[i] = 4.0f/(1.0f + x*x); 
	}
}

void initForPi(float *f, int32_t n){
	dim3 blocks  = dim3((n-1)/BLOCK_SIZE + 1);
	dim3 threads = dim3(BLOCK_SIZE);
	SAFE_KERNEL_CALL( (initKernel<<<blocks, threads>>>(f, n)) );
}

__global__ void sumKernel(float *in, float *out, int32_t n){
	// ????????????????	
}

float reduceRecursion(float *f, int32_t n){
	float result = 0.0;
	if (n > 2 && (n % 2) == 0){
		// ??????????????????????????????
	}
	else{
		float *fHost = (float *)malloc(n*sizeof(float));
		SAFE_CALL( cudaMemcpy(fHost, f, n*sizeof(float), cudaMemcpyDeviceToHost) );
		for (int i=0; i<n; ++i){
			result += fHost[i];
		}
		free(fHost);
	}

	return result;
}

float reduce(float *f, int32_t n){
	return reduceRecursion(f, n);
}

