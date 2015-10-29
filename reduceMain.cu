#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include <limits>
#define _USE_MATH_DEFINES
#include <math.h>
#include "inttypes.h"
#include "cudaErrorHadling.h"

float reduce(float *data, int32_t n);
void initForPi(float *data, int32_t n);

const char * const printMemorySize(size_t bytes){
    char inches[] = {' ', 'K', 'M', 'G', 'T'};
    double sz = bytes;

    int inch = 0;
    for (; sz > 512 && inch < 5; ++inch){
        sz /= 1024;
    }

    static char ret[64];
    sprintf(ret, "%.2f %cB", sz, inches[inch]);

    return ret;
}

float timer(){
    static clock_t timer = 0;
    if (!timer){
        timer = clock();

        return 0;
    }
    
    clock_t current = clock();
    float ret = ((float)(current - timer))/CLOCKS_PER_SEC;

    timer = current;
    return ret;
}

bool ourRequirementsPassed(const cudaDeviceProp & devProp){
    return devProp.major >= 1;
}

int selectCUDADevice(){
    int deviceCount = 0, suitableDevice = -1;
    cudaDeviceProp devProp;   
    cudaGetDeviceCount( &deviceCount );
    std::cout << "Found "<< deviceCount << " devices: \n";

    for (int device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties ( &devProp, device );

        std::cout << "Device: " << device                                               << std::endl;
        std::cout << "   Compute capability: " << devProp.major << "." << devProp.minor << std::endl;
        std::cout << "   Name: " << devProp.name                                        << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "   Total Global Memory: " << printMemorySize(devProp.totalGlobalMem)               << std::endl;
        std::cout << "   Shared Memory Per Block: " << printMemorySize(devProp.sharedMemPerBlock)        << std::endl;
        std::cout << "   Total Const Memory: " << printMemorySize(devProp.totalConstMem)        << std::endl;
        std::cout << "   L2 Cache size: " << printMemorySize(devProp.l2CacheSize)        << std::endl;
        std::cout << "   Memory bus width: " << printMemorySize(devProp.memoryBusWidth/8)        << std::endl;
        std::cout << "   Memory frequency: " << devProp.memoryClockRate << " kHz"       << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "   Multiprocessors: " << devProp.multiProcessorCount        << std::endl;
        std::cout << "   Clock rate: " << devProp.clockRate << " kHz"       << std::endl;
        std::cout << "   Warp Size: " << devProp.warpSize        << std::endl;
        std::cout << "   Max grid size: " << "(" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", "  << devProp.maxGridSize[2] << ")"      << std::endl;
        std::cout << "   Max block size: " << "(" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", "  << devProp.maxThreadsDim[2] << ")"      << std::endl;
        std::cout << "   Max threads per multiprocessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "   Max threads per block: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "   Registers per block: " << devProp.regsPerBlock << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::endl;

        if(suitableDevice < 0 && ourRequirementsPassed(devProp)){
            suitableDevice = device;
        }
    }
    return suitableDevice;
}

void initializeRandomArray(float *array, int length){
    for(int i =0; i < length; ++i){
        array[i] = ((float)(rand()%10))/length;
    }
}

int main(int argc, char *argv[]){
    //------------- Variables -----------
        int n = 1024*1024*8;
        cudaEvent_t start, stop;
        float timeGPU = 0.0;

        size_t nb = n*sizeof(float);
        float *aDev  = NULL;
        float sumDevice = 0.0;
    //-----------------------------------

    //--------- Command line -----------
        if(argc > 1){
            int tmp = atoi(argv[1]);
            if (tmp > 1){
                n = atoi(argv[1]);
            }
        }
    //----------------------------------

    //-------- Select device -----------
        int device = selectCUDADevice();  
        if(device == -1) {
            std::cout << "Can not find suitable device" << std::endl;
            return EXIT_FAILURE;
        }
        SAFE_CALL(cudaSetDevice(device));
    //-----------------------------------

    //----- GPU memory allocation and initialization -------
        SAFE_CALL( cudaMalloc((void**)&aDev, nb) );
        initForPi(aDev, n);
    //------------------------------------------------------

    //------ Create CUDA events ----------------------------
        SAFE_CALL( cudaEventCreate(&start) );
        SAFE_CALL( cudaEventCreate(&stop)  );
    //------------------------------------------------------

    //------ Calculation on GPU first way --------------
        SAFE_CALL( cudaEventRecord(start, 0) );

        float sum = reduce(aDev, n);

        SAFE_CALL( cudaEventRecord(stop, 0) );
        SAFE_CALL( cudaEventSynchronize(stop) );
        SAFE_CALL( cudaEventElapsedTime(&timeGPU, start, stop) );      
    //--------------------------------------

    double pi = sum/n;
    printf("~Pi = %e\tpi - ~Pi = %e\n", pi, M_PI-pi);
    printf("Processing time on GPU: %4.8f s\n", timeGPU/1000.0);

    getchar();
    
    return EXIT_SUCCESS;
}

