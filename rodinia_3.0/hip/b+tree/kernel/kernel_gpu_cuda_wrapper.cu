#include "hip/hip_runtime.h"
#define PROFILING 1
#ifdef PROFILING
#include "RDTimer.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"								// (in main program directory)			needed to recognized input variables

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/cuda/cuda.h"					// (in path specified to compiler)	needed by for device functions
//#include "../util/timer/timer.h"					// (in path specified to compiler)	needed by timer

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_gpu_cuda.cu"						// (in current directory)	GPU kernel, cannot include with header file because of complications with passing of constant memory variables

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_wrapper.h"				// (in current directory)

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200


void 
kernel_gpu_cuda_wrapper(record *records,
						long records_mem,
						knode *knodes,
						long knodes_elem,
						long knodes_mem,

						int order,
						long maxheight,
						int count,

						long *currKnode,
						long *offset,
						int *keys,
						record *ans)
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

//	time0 = get_time();

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
	numBlocks = count;									// max # of blocks can be 65,535
	int threadsPerBlock;
	threadsPerBlock = order < 1024 ? order : 1024;

	printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);

//	time1 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY				(MALLOC)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN
	//====================================================================================================100

	//==================================================50
	//	recordsD
	//==================================================50


/* malloc time-start */ 
#ifdef PROFILING
	float alloc_t1, cpu_to_gpu_t1,kernel_t1,gpu_to_cpu_t1;
        SimplePerfSerializer* serializeTime = new SimplePerfSerializer("./b+tree_1" );

        RDTimerCPU* rdtimercpu = new RDTimerCPU();

        rdtimercpu->Reset("Malloc Time");
        rdtimercpu->Start();
#endif

	record *recordsD;
	hipMalloc((void**)&recordsD, records_mem);
	checkCUDAError("hipMalloc  recordsD");

	//==================================================50
	//	knodesD
	//==================================================50

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
	//	keysD
	//==================================================50

	int *keysD;
	hipMalloc((void**)&keysD, count*sizeof(int));
	checkCUDAError("hipMalloc  keysD");

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansD
	//==================================================50

	record *ansD;
	hipMalloc((void**)&ansD, count*sizeof(record));
	checkCUDAError("hipMalloc ansD");

//	time2 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	recordsD
	//==================================================50

/* malloc time-stop, cpu-gpu transfer start */ 
#ifdef PROFILING
        alloc_t1 = rdtimercpu->Stop();
        serializeTime->Serialize(rdtimercpu);
        rdtimercpu->Reset("CPU to GPU Transfer Time");
        rdtimercpu->Start();
#endif

	hipMemcpy(recordsD, records, records_mem, hipMemcpyHostToDevice);
	checkCUDAError("hipMalloc hipMemcpy memD");

	//==================================================50
	//	knodesD
	//==================================================50

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
	//	keysD
	//==================================================50

	hipMemcpy(keysD, keys, count*sizeof(int), hipMemcpyHostToDevice);
	checkCUDAError("hipMalloc hipMemcpy keysD");

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansD
	//==================================================50

	hipMemcpy(ansD, ans, count*sizeof(record), hipMemcpyHostToDevice);
	checkCUDAError("hipMalloc hipMemcpy ansD");

/*cpu-gpu transfer-stop, kernel exec-start */
#ifdef PROFILING
    cpu_to_gpu_t1 =  rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("COMPUTE:Kernel Execution Time");
    hipDeviceSynchronize();
    rdtimercpu->Start();
#endif
//	time3 = get_time();

	//======================================================================================================================================================150
	// findK kernel
	//======================================================================================================================================================150

	hipLaunchKernel(findK, dim3(numBlocks), dim3(threadsPerBlock), 0, 0, 	maxheight,

											knodesD,
											knodes_elem,

											recordsD,

											currKnodeD,
											offsetD,
											keysD,
											ansD);
	hipDeviceSynchronize();
	checkCUDAError("findK");

//	time4 = get_time();
/* kernel exec-stop, gpu-cpu transfer-start */
#ifdef PROFILING
    kernel_t1 = rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("GPU to CPU Transfer Time");
    rdtimercpu->Start();
#endif


	//======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansD
	//==================================================50

	hipMemcpy(ans, ansD, count*sizeof(record), hipMemcpyDeviceToHost);
	checkCUDAError("hipMemcpy ansD");

//	time5 = get_time();

/* gpu-cpu transfer- stop */ 
#ifdef PROFILING        
    gpu_to_cpu_t1= rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
#endif

	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150

	hipFree(recordsD);
	hipFree(knodesD);

	hipFree(currKnodeD);
	hipFree(offsetD);
	hipFree(keysD);
	hipFree(ansD);

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

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200

/* over all stop print all times delete timers */
#ifdef PROFILING    
      printf("time CPU to GPU memory copy = %lfs\n", cpu_to_gpu_t1);
      printf("time GPU to CPU memory copy back = %lfs\n", gpu_to_cpu_t1);
      printf("time GPU malloc = %lfs\n", alloc_t1);
      printf("time kernel = %lfs\n", kernel_t1);
          delete rdtimercpu;
          delete serializeTime;
 #endif

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

#ifdef __cplusplus
}
#endif
