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

/* ---------------- first way ----------------------- */
__global__ void sumKernel1(float *in, float *out, int32_t n){
	int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n){
		out[i] = in[i] + in[i+n];
	}
}

float reduceRecursion1(float *f, int32_t n){
	float result = 0.0;
	if (n > 2 && (n % 2) == 0){
		float *partialSums;
		n /= 2;
		SAFE_CALL( cudaMalloc(&partialSums, n*sizeof(float)) );

		dim3 blocks  = dim3((n-1)/BLOCK_SIZE + 1);
		dim3 threads = dim3(BLOCK_SIZE);

		//printf("n = %d\tblocks = %d\n", n, blocks.x);

		SAFE_KERNEL_CALL( (sumKernel1<<<blocks, threads>>>(f, partialSums, n)) );
		result = reduceRecursion1(partialSums, n);

		SAFE_CALL( cudaFree(partialSums) );
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
float reduce1(float *f, int32_t n){
	return reduceRecursion1(f, n);
}
/***********************************************************/


/* ---------------- second way ----------------------- */
__global__ void reduceKernel(float *in, float *out){
	volatile __shared__ float data[BLOCK_SIZE];
	int32_t tid = threadIdx.x;
	int32_t i   = 2*blockIdx.x*blockDim.x + threadIdx.x;

	// sum two elements into the MP-th shared memory
	data[tid] = in[i] + in[i + blockDim.x];
	__syncthreads();

	for (int32_t j = blockDim.x/2; j > 32; j >>=1){
		if (tid < j){
			data[tid] += data[tid + j];
		}
		__syncthreads();
	}

	if (tid < 32){//unroll last iterations
		data[tid] += data[tid + 32];
		data[tid] += data[tid + 16];
		data[tid] += data[tid + 8 ];
		data[tid] += data[tid + 4 ];
		data[tid] += data[tid + 2 ];
		data[tid] += data[tid + 1 ];
	}

	if (tid == 0){// first thread writes block result
		out[blockIdx.x] = data[0];
	}
}

float sumVector1(float *f, int32_t n){
	float sum = 0;
	float c = 0;
	for (int i=0; i<n; ++i){
		float y = f[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
	}

	return sum;
}

float sumVector(float *f, int32_t n){
	float sum = 0;
	for (int i=0; i<n; ++i){
        sum += f[i];
	}

	return sum;
}

float reduce(float *f, int32_t n){
	int32_t blocks  = n / (2*BLOCK_SIZE);
	float result = 0;

	if (blocks > BLOCK_SIZE && (n % 2*BLOCK_SIZE) == 0){
		float *partialSums = NULL;
		SAFE_CALL( cudaMalloc(&partialSums, sizeof(float)*blocks) );
		
		SAFE_KERNEL_CALL( (reduceKernel<<<blocks, BLOCK_SIZE >>>(f, partialSums)) );
		result = reduce(partialSums, blocks);
		
		SAFE_CALL( cudaFree(partialSums) );
	}
	else{
		float *fHost = (float *)malloc(n*sizeof(float));
		SAFE_CALL( cudaMemcpy(fHost, f, n*sizeof(float), cudaMemcpyDeviceToHost) );
		result = sumVector1(fHost, n);
		free(fHost);
	}

	return result;
}
