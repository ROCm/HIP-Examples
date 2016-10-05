#include "hip/hip_runtime.h"
#define PROFILING 1
#ifdef PROFILING
#include "RDTimer.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

//========================================================================================================================================================================================================200
//	INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"									// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/cuda/cuda.h"							// (in library path specified to compiler)	needed by for device functions
//#include "../util/timer/timer.h"						// (in library path specified to compiler)	needed by timer

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_2.cu"						// (in the current directory)	GPU kernel, cannot include with header file because of complications with passing of constant memory variables

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_wrapper_2.h"				// (in the current directory)

//========================================================================================================================================================================================================200
//	FUNCTION
//========================================================================================================================================================================================================200
void 
kernel_gpu_cuda_wrapper_2(	knode *knodes,
							long knodes_elem,
							long knodes_mem,

							int order,
							long maxheight,
							int count,

							long *currKnode,
							long *offset,
							long *lastKnode,
							long *offset_2,
							int *start,
							int *end,
							int *recstart,
							int *reclength)
{

	//======================================================================================================================================================150
	//	CPU VARIABLES
	//======================================================================================================================================================150

	// timer
/*	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;*/

	//time0 = get_time();

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	hipDeviceSynchronize();

	//====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	int numBlocks;
	numBlocks = count;
	int threadsPerBlock;
	threadsPerBlock = order < 1024 ? order : 1024;

	printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);

//	time1 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY				MALLOC
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN
	//====================================================================================================100

	//==================================================50
	//	knodesD
	//==================================================50

/* malloc time-start */ 
#ifdef PROFILING
	float alloc_t2, cpu_to_gpu_t2,kernel_t2,gpu_to_cpu_t2;
        SimplePerfSerializer* serializeTime = new SimplePerfSerializer("./b+tree_2");
        RDTimerCPU* rdtimercpu = new RDTimerCPU();

        rdtimercpu->Reset("Malloc Time");
        rdtimercpu->Start();
#endif
	knode *knodesD;
	hipMalloc((void**)&knodesD, knodes_mem);
	checkCUDAError("hipMalloc  recordsD");

	//==================================================50
	//	currKnodeD
	//==================================================50

	long *currKnodeD;
	hipMalloc((void**)&currKnodeD, count*sizeof(long));
	checkCUDAError("hipMalloc  currKnodeD");

	//==================================================50
	//	offsetD
	//==================================================50

	long *offsetD;
	hipMalloc((void**)&offsetD, count*sizeof(long));
	checkCUDAError("hipMalloc  offsetD");

	//==================================================50
	//	lastKnodeD
	//==================================================50

	long *lastKnodeD;
	hipMalloc((void**)&lastKnodeD, count*sizeof(long));
	checkCUDAError("hipMalloc  lastKnodeD");

	//==================================================50
	//	offset_2D
	//==================================================50

	long *offset_2D;
	hipMalloc((void**)&offset_2D, count*sizeof(long));
	checkCUDAError("hipMalloc  offset_2D");

	//==================================================50
	//	startD
	//==================================================50

	int *startD;
	hipMalloc((void**)&startD, count*sizeof(int));
	checkCUDAError("hipMalloc startD");

	//==================================================50
	//	endD
	//==================================================50

	int *endD;
	hipMalloc((void**)&endD, count*sizeof(int));
	checkCUDAError("hipMalloc endD");

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50

	int *ansDStart;
	hipMalloc((void**)&ansDStart, count*sizeof(int));
	checkCUDAError("hipMalloc ansDStart");

	//==================================================50
	//	ansDLength
	//==================================================50

	int *ansDLength;
	hipMalloc((void**)&ansDLength, count*sizeof(int));
	checkCUDAError("hipMalloc ansDLength");

//	time2 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN
	//====================================================================================================100

	//==================================================50
	//	knodesD
	//==================================================50

/* malloc time-stop, cpu-gpu transfer start */ 
#ifdef PROFILING
        alloc_t2 = rdtimercpu->Stop();
        serializeTime->Serialize(rdtimercpu);
        rdtimercpu->Reset("CPU to GPU Transfer Time");
        rdtimercpu->Start();
#endif

	hipMemcpy(knodesD, knodes, knodes_mem, hipMemcpyHostToDevice);
	checkCUDAError("hipMalloc hipMemcpy memD");

	//==================================================50
	//	currKnodeD
	//==================================================50

	hipMemcpy(currKnodeD, currKnode, count*sizeof(long), hipMemcpyHostToDevice);
	checkCUDAError("hipMalloc hipMemcpy currKnodeD");

	//==================================================50
	//	offsetD
	//==================================================50

	hipMemcpy(offsetD, offset, count*sizeof(long), hipMemcpyHostToDevice);
	checkCUDAError("hipMalloc hipMemcpy offsetD");

	//==================================================50
	//	lastKnodeD
	//==================================================50

	hipMemcpy(lastKnodeD, lastKnode, count*sizeof(long), hipMemcpyHostToDevice);
	checkCUDAError("hipMalloc hipMemcpy lastKnodeD");

	//==================================================50
	//	offset_2D
	//==================================================50

	hipMemcpy(offset_2D, offset_2, count*sizeof(long), hipMemcpyHostToDevice);
	checkCUDAError("hipMalloc hipMemcpy offset_2D");

	//==================================================50
	//	startD
	//==================================================50

	hipMemcpy(startD, start, count*sizeof(int), hipMemcpyHostToDevice);
	checkCUDAError("hipMemcpy startD");

	//==================================================50
	//	endD
	//==================================================50

	hipMemcpy(endD, end, count*sizeof(int), hipMemcpyHostToDevice);
	checkCUDAError("hipMemcpy endD");

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50

	hipMemcpy(ansDStart, recstart, count*sizeof(int), hipMemcpyHostToDevice);
	checkCUDAError("hipMemcpy ansDStart");

	//==================================================50
	//	ansDLength
	//==================================================50

	hipMemcpy(ansDLength, reclength, count*sizeof(int), hipMemcpyHostToDevice);
	checkCUDAError("hipMemcpy ansDLength");

//	time3 = get_time();

/*cpu-gpu transfer-stop, kernel exec-start */
#ifdef PROFILING
    cpu_to_gpu_t2 =  rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("COMPUTE:Kernel Execution Time");
    rdtimercpu->Start();
#endif
	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	// [GPU] findRangeK kernel
	hipLaunchKernel(findRangeK, dim3(numBlocks), dim3(threadsPerBlock), 0, 0, 	maxheight,
												knodesD,
												knodes_elem,

												currKnodeD,
												offsetD,
												lastKnodeD,
												offset_2D,
												startD,
												endD,
												ansDStart,
												ansDLength);
	hipDeviceSynchronize();
	checkCUDAError("findRangeK");

/* kernel exec-stop, gpu-cpu transfer-start */
#ifdef PROFILING
    kernel_t2 = rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("GPU to CPU Transfer Time");
    rdtimercpu->Start();
#endif

//	time4 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50

	hipMemcpy(recstart, ansDStart, count*sizeof(int), hipMemcpyDeviceToHost);
	checkCUDAError("hipMemcpy ansDStart");

	//==================================================50
	//	ansDLength
	//==================================================50

	hipMemcpy(reclength, ansDLength, count*sizeof(int), hipMemcpyDeviceToHost);
	checkCUDAError("hipMemcpy ansDLength");

//	time5 = get_time();

/* gpu-cpu transfer- stop */ 
#ifdef PROFILING        
    gpu_to_cpu_t2= rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);

	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150


/* print all times delete timers */
    
      printf("time CPU to GPU memory copy = %lfs\n", cpu_to_gpu_t2);
      printf("time GPU to CPU memory copy back = %lfs\n", gpu_to_cpu_t2);
      printf("time GPU malloc = %lfs\n", alloc_t2);
      printf("time kernel = %lfs\n", kernel_t2);
          delete rdtimercpu;
          delete serializeTime;
#endif
	hipFree(knodesD);

	hipFree(currKnodeD);
	hipFree(offsetD);
	hipFree(lastKnodeD);
	hipFree(offset_2D);
	hipFree(startD);
	hipFree(endD);
	hipFree(ansDStart);
	hipFree(ansDLength);

//	time6 = get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

/*	printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

	printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",	(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: ALO\n", 					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",					(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time6-time0) * 100);

	printf("%15.12f s, %15.12f % : GPU: KERNEL\n",						(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time6-time0) * 100);

	printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",				(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: FRE\n", 					(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time6-time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) (time6-time0) / 1000000);
*/
}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

#ifdef __cplusplus
}
#endif
