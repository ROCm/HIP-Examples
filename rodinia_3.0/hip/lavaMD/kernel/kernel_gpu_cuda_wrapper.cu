#include "hip/hip_runtime.h"
//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200
#define PROFILING 1
#ifdef PROFILING
#include "RDTimer.h"
#endif
//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./../main.h"								// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./../util/device/device.h"				// (in library path specified to compiler)	needed by for device functions
//#include "./../util/timer/timer.h"					// (in library path specified to compiler)	needed by timer

//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_wrapper.h"				// (in the current directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_gpu_cuda.cu"						// (in the current directory)	GPU kernel, cannot include with header file because of complications with passing of constant memory variables

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_cuda_wrapper(par_str par_cpu,
						dim_str dim_cpu,
						box_str* box_cpu,
						FOUR_VECTOR* rv_cpu,
						fp* qv_cpu,
						FOUR_VECTOR* fv_cpu)
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

/* overall time - start */
#ifdef PROFILING
	float alloc_t, cpu_to_gpu_t,kernel_t,gpu_to_cpu_t,overall_cpu_t;
        RDTimerCPU* rdtimerOverallCpu = new RDTimerCPU();
        rdtimerOverallCpu->Reset("Overall CPU Time");
        rdtimerOverallCpu->Start();
#endif

	hipDeviceSynchronize();

	//====================================================================================================100
	//	VARIABLES
	//====================================================================================================100

	box_str* d_box_gpu;
	FOUR_VECTOR* d_rv_gpu;
	fp* d_qv_gpu;
	FOUR_VECTOR* d_fv_gpu;

	dim3 threads;
	dim3 blocks;

	//====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	blocks.x = dim_cpu.number_boxes;
	blocks.y = 1;
	threads.x = NUMBER_THREADS;											// define the number of threads in the block
	threads.y = 1;

	//time1 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY				(MALLOC)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

/* malloc time-start */ 
#ifdef PROFILING
        SimplePerfSerializer* serializeTime = new SimplePerfSerializer( "./lavaMD" );
        RDTimerCPU* rdtimercpu = new RDTimerCPU();

        rdtimercpu->Reset("Malloc Time");
        rdtimercpu->Start();
#endif
	//==================================================50
	//	boxes
	//==================================================50

	hipMalloc(	(void **)&d_box_gpu, 
				dim_cpu.box_mem);

	//==================================================50
	//	rv
	//==================================================50

	hipMalloc(	(void **)&d_rv_gpu, 
				dim_cpu.space_mem);

	//==================================================50
	//	qv
	//==================================================50

	hipMalloc(	(void **)&d_qv_gpu, 
				dim_cpu.space_mem2);

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	hipMalloc(	(void **)&d_fv_gpu, 
				dim_cpu.space_mem);

	//time2 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150
#ifdef PROFILING
        alloc_t = rdtimercpu->Stop();
        serializeTime->Serialize(rdtimercpu);
        rdtimercpu->Reset("CPU to GPU Transfer Time");
        rdtimercpu->Start();
#endif
	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

	hipMemcpy(	d_box_gpu, 
				box_cpu,
				dim_cpu.box_mem, 
				hipMemcpyHostToDevice);

	//==================================================50
	//	rv
	//==================================================50

	hipMemcpy(	d_rv_gpu,
				rv_cpu,
				dim_cpu.space_mem,
				hipMemcpyHostToDevice);

	//==================================================50
	//	qv
	//==================================================50

	hipMemcpy(	d_qv_gpu,
				qv_cpu,
				dim_cpu.space_mem2,
				hipMemcpyHostToDevice);

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	hipMemcpy(	d_fv_gpu, 
				fv_cpu, 
				dim_cpu.space_mem, 
				hipMemcpyHostToDevice);

	//time3 = get_time();

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150
/*cpu-gpu transfer-stop, kernel exec-start */
#ifdef PROFILING
    cpu_to_gpu_t =  rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("COMPUTE:Kernel Execution Time");
    rdtimercpu->Start();
#endif
	// launch kernel - all boxes
	hipLaunchKernel(kernel_gpu_cuda, dim3(blocks), dim3(threads), 0, 0, 	par_cpu,
											dim_cpu,
											d_box_gpu,
											d_rv_gpu,
											d_qv_gpu,
											d_fv_gpu);
/* kernel exec-stop, gpu-cpu transfer-start */
#ifdef PROFILING
    kernel_t = rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("GPU to CPU Transfer Time");
    rdtimercpu->Start();
#endif
	//checkCUDAError("Start");
	hipDeviceSynchronize();

	//time4 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)
	//======================================================================================================================================================150

	hipMemcpy(	fv_cpu, 
				d_fv_gpu, 
				dim_cpu.space_mem, 
				hipMemcpyDeviceToHost);

	//time5 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150
#ifdef PROFILING        
    gpu_to_cpu_t= rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    overall_cpu_t =  rdtimerOverallCpu->Stop();
    serializeTime->Serialize(rdtimerOverallCpu);
      printf("time CPU to GPU memory copy = %lfs\n", cpu_to_gpu_t);
      printf("time GPU to CPU memory copy back = %lfs\n", gpu_to_cpu_t);
      printf("time GPU malloc = %lfs\n", alloc_t);
      printf("time kernel = %lfs\n", kernel_t);
      printf("Overall CPU time = %lfs\n", overall_cpu_t);
          delete rdtimercpu;
          delete serializeTime;
          delete rdtimerOverallCpu;

 #endif

	hipFree(d_rv_gpu);
	hipFree(d_qv_gpu);
	hipFree(d_fv_gpu);
	hipFree(d_box_gpu);

	//time6 = get_time();

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
