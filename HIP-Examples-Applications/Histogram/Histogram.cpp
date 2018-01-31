/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#include "Histogram.hpp"

#include <math.h>

#define LINEAR_MEM_ACCESS

#define BIN_SIZE 256

/**
 * @brief   Calculates block-histogram bin whose bin size is 256
 * @param   data  input data pointer
 * @param   sharedArray shared array for thread-histogram bins
 * @param   binResult block-histogram array
 */
__global__
void histogram256(hipLaunchParm lp,
                        unsigned int* data,
                        unsigned int* binResult)
{
    HIP_DYNAMIC_SHARED(unsigned char, sharedArray);
    size_t localId = hipThreadIdx_x;
    size_t globalId = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
    size_t groupId = hipBlockIdx_x;
    size_t groupSize = hipBlockDim_x;
    int offSet1 = localId & 31;    
    int offSet2 = 4 * offSet1;      //which element to access in one bank.
    int offSet3 = localId >> 5;     //bank number
    /* initialize shared array to zero */
    uchar4 * input = (uchar4*)sharedArray;
    for(int i = 0; i < 64; ++i)
        input[groupSize * i + localId] = make_uchar4(0,0,0,0);

    __syncthreads();


    /* calculate thread-histograms */
	//128 accumulations per thread
	for(int i = 0; i < 128; i++)
    {
#ifdef LINEAR_MEM_ACCESS
       uint value =  data[groupId * (groupSize * (BIN_SIZE/2)) + i * groupSize + localId]; 
#else
       uint  value = data[globalId + i*4096];

#endif // LINEAR_MEM_ACCESS
	   sharedArray[value * 128 + offSet2 + offSet3]++;
    }  
    __syncthreads();
    
    /* merge all thread-histograms into block-histogram */

	uint4 binCount;
	uint result;
	uchar4 binVal;	            //Introduced uint4 for summation to avoid overflows
	uint4 binValAsUint;
	for(int i = 0; i < BIN_SIZE / groupSize; ++i)
    {
        int passNumber = BIN_SIZE / 2 * 32 * i +  localId * 32 ;
		binCount = make_uint4(0,0,0,0);
		result= 0;
        for(int j = 0; j < 32; ++j)
		{
			int bankNum = (j + offSet1) & 31;   // this is bank number
            binVal = input[passNumber  +bankNum];

            binValAsUint.x = (unsigned int)binVal.x;
            binValAsUint.y = (unsigned int)binVal.y;
            binValAsUint.z = (unsigned int)binVal.z;
            binValAsUint.w = (unsigned int)binVal.w;

            binCount.x += binValAsUint.x;
            binCount.y += binValAsUint.y;
            binCount.z += binValAsUint.z;
            binCount.w += binValAsUint.w;

		}
        result = binCount.x + binCount.y + binCount.z + binCount.w;
        binResult[groupId * BIN_SIZE + groupSize * i + localId ] = result;
	}
}

int
Histogram::calculateHostBin()
{
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            hostBin[data[i * width + j]]++;
        }
    }

    return SDK_SUCCESS;
}


int
Histogram::setupHistogram()
{
    int i = 0;

    data = (unsigned int *)malloc(sizeof(unsigned int) * width * height);

    for(i = 0; i < width * height; i++)
    {
        data[i] = rand() % (unsigned int)(binSize);
    }

    hostBin = (unsigned int*)malloc(binSize * sizeof(unsigned int));
    CHECK_ALLOCATION(hostBin, "Failed to allocate host memory. (hostBin)");

    memset(hostBin, 0, binSize * sizeof(unsigned int));

    deviceBin = (unsigned int*)malloc(binSize * sizeof(unsigned int));
    CHECK_ALLOCATION(deviceBin, "Failed to allocate host memory. (deviceBin)");
    midDeviceBin = (unsigned int*)malloc(sizeof(unsigned int) * binSize * subHistgCnt);

    memset(deviceBin, 0, binSize * sizeof(unsigned int));


    if(scalar && vector)//if both options are specified
    {
        std::cout<<"Ignoring --scalar and --vector option and using the default vector width of 4"<<std::endl;
    }

    else if(scalar)
    {
        vectorWidth = 1;
    }
    else if(vector)
    {
        vectorWidth = 4;
    }
    else //if no option is specified.
    {
        vectorWidth = 1;
    }

    if(!sampleArgs->quiet)
    {
        if(vectorWidth == 4)
        {
            std::cout<<"Selecting scalar kernel\n"<<std::endl;
        }
        else
        {
            std::cout<<"Selecting vector kernel\n"<<std::endl;
        }
    }

    return SDK_SUCCESS;
}

int
Histogram::setupHIP(void)
{
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    return SDK_SUCCESS;
}



int
Histogram::runKernels(void)
{

    hipEvent_t start, stop;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;

    groupSize = 128;
    globalThreads = (width * height) / (GROUP_ITERATIONS);

    localThreads = groupSize;


    hipHostMalloc((void**)&dataBuf,sizeof(unsigned int) * width * height, hipHostMallocDefault);
    unsigned int *din;
    hipHostGetDevicePointer((void**)&din, dataBuf,0);
    hipMemcpy(din, data,sizeof(unsigned int) * width * height, hipMemcpyHostToDevice);

    subHistgCnt = (width * height) / (groupSize * groupIterations);

    hipHostMalloc((void**)&midDeviceBinBuf,sizeof(unsigned int) * binSize * subHistgCnt, hipHostMallocDefault);

    hipEventRecord(start, NULL);

    hipLaunchKernel(histogram256,
                    dim3(globalThreads/localThreads),
                    dim3(localThreads),
                    groupSize * binSize * sizeof(unsigned char), 0,
                    dataBuf ,midDeviceBinBuf);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf ("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);

    hipMemcpy(midDeviceBin, midDeviceBinBuf,sizeof(unsigned int) * binSize * subHistgCnt, hipMemcpyDeviceToHost);
        //printArray<unsigned int>("midDeviceBin", midDeviceBin, sizeof(unsigned int) * binSize * subHistgCnt, 1);
    // Clear deviceBin array
    memset(deviceBin, 0, binSize * sizeof(unsigned int));

    // Calculate final histogram bin
    for(int i = 0; i < subHistgCnt; ++i)
    {
        for(int j = 0; j < binSize; ++j)
        {
            deviceBin[j] += midDeviceBin[i * binSize + j];
        }
    }

    return SDK_SUCCESS;
}

int
Histogram::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize())
    {
        return SDK_FAILURE;
    }

    Option* width_option = new Option;
    CHECK_ALLOCATION(width_option, "Memory allocation error.\n");

    width_option->_sVersion = "x";
    width_option->_lVersion = "width";
    width_option->_description = "Width of the input";
    width_option->_type = CA_ARG_INT;
    width_option->_value = &width;

    sampleArgs->AddOption(width_option);
    delete width_option;

    Option* height_option = new Option;
    CHECK_ALLOCATION(height_option, "Memory allocation error.\n");

    height_option->_sVersion = "y";
    height_option->_lVersion = "height";
    height_option->_description = "Height of the input";
    height_option->_type = CA_ARG_INT;
    height_option->_value = &height;

    sampleArgs->AddOption(height_option);
    delete height_option;

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option, "Memory allocation error.\n");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    Option* scalar_option = new Option;
    CHECK_ALLOCATION(scalar_option, "Memory allocation error.\n");

    scalar_option->_sVersion = "";
    scalar_option->_lVersion = "scalar";
    scalar_option->_description =
        "Run scalar version of the kernel (--scalar and --vector options are mutually exclusive)";
    scalar_option->_type = CA_NO_ARGUMENT;
    scalar_option->_value = &scalar;

    sampleArgs->AddOption(scalar_option);
    delete scalar_option;

    Option* vector_option = new Option;
    CHECK_ALLOCATION(vector_option, "Memory allocation error.\n");

    vector_option->_sVersion = "";
    vector_option->_lVersion = "vector";
    vector_option->_description =
        "Run vector version of the kernel (--scalar and --vector options are mutually exclusive)";
    vector_option->_type = CA_NO_ARGUMENT;
    vector_option->_value = &vector;

    sampleArgs->AddOption(vector_option);
    delete vector_option;



    return SDK_SUCCESS;
}

int
Histogram::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        exit(0);
    }
    int status = 0;

    /* width must be multiples of binSize and
     * height must be multiples of groupSize
     */
    width = (width / binSize ? width / binSize: 1) * binSize;
    height = (height / groupSize ? height / groupSize: 1) * groupSize;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    status = setupHIP();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    status = setupHistogram();
    CHECK_ERROR(status, SDK_SUCCESS, "Sample Resource Setup Failed");

    sampleTimer->stopTimer(timer);

    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int
Histogram::run()
{

    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " <<
              iterations << " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    // Compute average kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer));

    if(!sampleArgs->quiet)
    {
        printArray<unsigned int>("deviceBin", deviceBin, binSize, 1);
    }

    return SDK_SUCCESS;
}

int
Histogram::verifyResults()
{
    if(sampleArgs->verify)
    {
        /**
         * Reference implementation on host device
         * calculates the histogram bin on host
         */
        calculateHostBin();
        printArray<unsigned int>("hostBin", hostBin, binSize, 1);
        // compare the results and see if they match
        bool result = true;
        for(int i = 0; i < binSize; ++i)
        {
            if(hostBin[i] != deviceBin[i])
            {
                result = false;
                break;
            }
        }

        if(result)
        {
            std::cout << "Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void Histogram::printStats()
{
    if(sampleArgs->timing)
    {
        // calculate total time
        double avgKernelTime = kernelTime/iterations;

        std::string strArray[5] =
        {
            "Width",
            "Height",
            "Setup Time(sec)",
            "Avg. Kernel Time (sec)",
            "Elements/sec"
        };
        std::string stats[5];

        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString(setupTime, std::dec);
        stats[3] = toString(avgKernelTime, std::dec);
        stats[4] = toString(((width*height)/avgKernelTime), std::dec);

        printStatistics(strArray, stats, 5);
    }
}

int Histogram::cleanup()
{

    // Releases HIP resources (Memory)


    hipFree(dataBuf);
    hipFree(midDeviceBinBuf);

    // Release program resources (input memory etc.)
    FREE(hostBin);
    FREE(deviceBin);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    int status = 0;
    // Create MonteCalroAsian object
    Histogram hipHistogram;

    // Initialization
    if(hipHistogram.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Parse command line options
    if(hipHistogram.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Setup
    status = hipHistogram.setup();
    if(status != SDK_SUCCESS)
    {
        return (status == SDK_EXPECTED_FAILURE)? SDK_SUCCESS : SDK_FAILURE;
    }

    // Run
    if(hipHistogram.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Verify
    if(hipHistogram.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Cleanup resources created
    if(hipHistogram.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Print performance statistics
    hipHistogram.printStats();

    return SDK_SUCCESS;
}
