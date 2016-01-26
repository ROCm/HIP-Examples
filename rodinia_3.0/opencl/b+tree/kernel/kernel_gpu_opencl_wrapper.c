// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <CL/cl.h>									// (in directory provided to compiler)		needed by OpenCL types and functions
#include <string.h>									// (in directory known to compiler)			needed by memset

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"								// (in directory provided here)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/opencl/opencl.h"					// (in directory provided here)
#include "../util/timer/timer.h"					// (in directory provided here)

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_opencl_wrapper.h"			// (in directory provided here)

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_opencl_wrapper(	record *records,
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
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;

	time0 = get_time();

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	// cudaThreadSynchronize();

	//====================================================================================================100
	//	COMMON VARIABLES
	//====================================================================================================100

	// common variables
	cl_int error;

	//====================================================================================================100
	//	GET PLATFORMS (Intel, AMD, NVIDIA, based on provided library), SELECT ONE
	//====================================================================================================100

	// Get the number of available platforms
	cl_uint num_platforms;
	error = clGetPlatformIDs(	0, 
								NULL, 
								&num_platforms);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Get the list of available platforms
	cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
	error = clGetPlatformIDs(	num_platforms, 
								platforms, 
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Select the 1st platform
	cl_platform_id platform = platforms[0];

	// Get the name of the selected platform and print it (if there are multiple platforms, choose the first one)
	char pbuf[100];
	error = clGetPlatformInfo(	platform, 
								CL_PLATFORM_VENDOR, 
								sizeof(pbuf), 
								pbuf, 
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	printf("Platform: %s\n", pbuf);

	//====================================================================================================100
	//	CREATE CONTEXT FOR THE PLATFORM
	//====================================================================================================100

	// Create context properties for selected platform
	cl_context_properties context_properties[3] = {	CL_CONTEXT_PLATFORM, 
													(cl_context_properties) platform, 
													0};

	// Create context for selected platform being GPU
	cl_context context;
	context = clCreateContextFromType(	context_properties, 
										CL_DEVICE_TYPE_GPU, 
										NULL, 
										NULL, 
										&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	GET DEVICES AVAILABLE FOR THE CONTEXT, SELECT ONE
	//====================================================================================================100

	// Get the number of devices (previousely selected for the context)
	size_t devices_size;
	error = clGetContextInfo(	context, 
								CL_CONTEXT_DEVICES, 
								0, 
								NULL, 
								&devices_size);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Get the list of devices (previousely selected for the context)
	cl_device_id *devices = (cl_device_id *) malloc(devices_size);
	error = clGetContextInfo(	context, 
								CL_CONTEXT_DEVICES, 
								devices_size, 
								devices, 
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Select the first device (previousely selected for the context) (if there are multiple devices, choose the first one)
	cl_device_id device;
	device = devices[0];

	// Get the name of the selected device (previousely selected for the context) and print it
	error = clGetDeviceInfo(device, 
							CL_DEVICE_NAME, 
							sizeof(pbuf), 
							pbuf, 
							NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	printf("Device: %s\n", pbuf);

	//====================================================================================================100
	//	CREATE COMMAND QUEUE FOR THE DEVICE
	//====================================================================================================100

	// Create a command queue
	cl_command_queue command_queue;
	command_queue = clCreateCommandQueue(	context, 
											device, 
											0, 
											&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	CREATE PROGRAM, COMPILE IT
	//====================================================================================================100

	// Load kernel source code from file
	const char *source = load_kernel_source("./kernel/kernel_gpu_opencl.cl");
	size_t sourceSize = strlen(source);

	// Create the program
	cl_program program = clCreateProgramWithSource(	context, 
													1, 
													&source, 
													&sourceSize, 
													&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	char clOptions[110];
	//  sprintf(clOptions,"-I../../src");                                                                                 
	sprintf(clOptions,"-I./../");

#ifdef DEFAULT_ORDER
	sprintf(clOptions + strlen(clOptions), " -DDEFAULT_ORDER=%d", DEFAULT_ORDER);
#endif


	// Compile the program
	error = clBuildProgram(	program, 
							1, 
							&device, 
							clOptions, 
							NULL, 
							NULL);
	// Print warnings and errors from compilation
	static char log[65536]; 
	memset(log, 0, sizeof(log));
	clGetProgramBuildInfo(	program, 
							device, 
							CL_PROGRAM_BUILD_LOG, 
							sizeof(log)-1, 
							log, 
							NULL);
	printf("-----OpenCL Compiler Output-----\n");
	if (strstr(log,"warning:") || strstr(log, "error:")) 
		printf("<<<<\n%s\n>>>>\n", log);
	printf("--------------------------------\n");
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Create kernel
	cl_kernel kernel;
	kernel = clCreateKernel(program, 
							"findK", 
							&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	time1 = get_time();

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//======================================================================================================================================================150
	//	GPU MEMORY				(MALLOC)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN
	//====================================================================================================100

	//==================================================50
	//	recordsD
	//==================================================50

	cl_mem recordsD;
	recordsD = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								records_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	knodesD
	//==================================================50

	cl_mem knodesD;
	knodesD = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								knodes_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	currKnodeD
	//==================================================50

	cl_mem currKnodeD;
	currKnodeD = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								count*sizeof(long), 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	offsetD
	//==================================================50

	cl_mem offsetD;
	offsetD = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								count*sizeof(long), 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	keysD
	//==================================================50

	cl_mem keysD;
	keysD = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								count*sizeof(long), 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	END
	//==================================================50

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansD
	//==================================================50

	cl_mem ansD;
	ansD = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								count*sizeof(record), 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	time2 = get_time();

	//==================================================50
	//	END
	//==================================================50

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	recordsD
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									recordsD,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									records_mem,			// size to be copied
									records,				// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									NULL);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	knodesD
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									knodesD,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									knodes_mem,				// size to be copied
									knodes,					// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									NULL);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	currKnodeD
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									currKnodeD,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(long),		// size to be copied
									currKnode,				// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									NULL);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	offsetD
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									offsetD,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(long),		// size to be copied
									offset,					// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									NULL);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	keysD
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									keysD,					// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(int),		// size to be copied
									keys,					// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									NULL);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	END
	//==================================================50

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansD
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									ansD,					// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(record),	// size to be copied
									ans,					// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									NULL);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	time3 = get_time();

	//==================================================50
	//	END
	//==================================================50

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//======================================================================================================================================================150
	// findK kernel
	//======================================================================================================================================================150

	//====================================================================================================100
	//	Execution Parameters
	//====================================================================================================100

	size_t local_work_size[1];
	local_work_size[0] = order < 1024 ? order : 1024;
	size_t global_work_size[1];
	global_work_size[0] = count * local_work_size[0];

	printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", (int)(global_work_size[0]/local_work_size[0]), (int)local_work_size[0]);

	//====================================================================================================100
	//	Kernel Arguments
	//====================================================================================================100

	clSetKernelArg(	kernel, 
					0, 
					sizeof(long), 
					(void *) &maxheight);
	clSetKernelArg(	kernel, 
					1, 
					sizeof(cl_mem), 
					(void *) &knodesD);
	clSetKernelArg(	kernel, 
					2, 
					sizeof(long), 
					(void *) &knodes_elem);
	clSetKernelArg(	kernel, 
					3, 
					sizeof(cl_mem), 
					(void *) &recordsD);

	clSetKernelArg(	kernel, 
					4, 
					sizeof(cl_mem), 
					(void *) &currKnodeD);
	clSetKernelArg(	kernel, 
					5, 
					sizeof(cl_mem), 
					(void *) &offsetD);
	clSetKernelArg(	kernel, 
					6, 
					sizeof(cl_mem), 
					(void *) &keysD);
	clSetKernelArg(	kernel, 
					7, 
					sizeof(cl_mem), 
					(void *) &ansD);

	//====================================================================================================100
	//	Kernel
	//====================================================================================================100

	error = clEnqueueNDRangeKernel(	command_queue, 
									kernel, 
									1, 
									NULL, 
									global_work_size, 
									local_work_size, 
									0, 
									NULL, 
									NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Wait for all operations to finish NOT SURE WHERE THIS SHOULD GO
	error = clFinish(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	time4 = get_time();

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansD
	//==================================================50

	error = clEnqueueReadBuffer(command_queue,				// The command queue.
								ansD,						// The image on the device.
								CL_TRUE,					// Blocking? (ie. Wait at this line until read has finished?)
								0,							// Offset. None in this case.
								count*sizeof(record),		// Size to copy.
								ans,						// The pointer to the image on the host.
								0,							// Number of events in wait list. Not used.
								NULL,						// Event wait list. Not used.
								NULL);						// Event object for determining status. Not used.
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	time5 = get_time();

	//==================================================50
	//	END
	//==================================================50

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150

	// Release kernels...
	clReleaseKernel(kernel);

	// Now the program...
	clReleaseProgram(program);

	// Clean up the device memory...
	clReleaseMemObject(recordsD);
	clReleaseMemObject(knodesD);

	clReleaseMemObject(currKnodeD);
	clReleaseMemObject(offsetD);
	clReleaseMemObject(keysD);
	clReleaseMemObject(ansD);

	// Flush the queue
	error = clFlush(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// ...and finally, the queue and context.
	clReleaseCommandQueue(command_queue);

	// ???
	clReleaseContext(context);

	time6 = get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

	printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",	(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: ALO\n", 					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",					(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time6-time0) * 100);

	printf("%15.12f s, %15.12f % : GPU: KERNEL\n",						(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time6-time0) * 100);

	printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",				(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: FRE\n", 					(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time6-time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) (time6-time0) / 1000000);

	//======================================================================================================================================================150
	//	END
	//======================================================================================================================================================150

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
