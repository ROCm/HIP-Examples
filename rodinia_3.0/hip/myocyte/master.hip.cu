#include "hip/hip_runtime.h"
//=====================================================================
//	MAIN FUNCTION
//=====================================================================

void master(fp timeinst,
					fp* initvalu,
					fp* parameter,
					fp* finavalu,
					fp* com,

					fp* d_initvalu,
					fp* d_finavalu,
					fp* d_params,
					fp* d_com){

	//=====================================================================
	//	VARIABLES
	//=====================================================================

	// counters
	int i;

	// offset pointers
	int initvalu_offset_ecc;																// 46 points
	int initvalu_offset_Dyad;															// 15 points
	int initvalu_offset_SL;																// 15 points
	int initvalu_offset_Cyt;																// 15 poitns

	// cuda
	dim3 threads;
	dim3 blocks;

	//=====================================================================
	//	execute ECC&CAM kernel - it runs ECC and CAMs in parallel
	//=====================================================================

	int d_initvalu_mem;
	d_initvalu_mem = EQUATIONS * sizeof(fp);
	int d_finavalu_mem;
	d_finavalu_mem = EQUATIONS * sizeof(fp);
	int d_params_mem;
	d_params_mem = PARAMETERS * sizeof(fp);
	int d_com_mem;
	d_com_mem = 3 * sizeof(fp);

	hipMemcpy(d_initvalu, initvalu, d_initvalu_mem, hipMemcpyHostToDevice);
	hipMemcpy(d_params, parameter, d_params_mem, hipMemcpyHostToDevice);

	threads.x = NUMBER_THREADS;
	threads.y = 1;
	blocks.x = 2;
	blocks.y = 1;
	hipLaunchKernel(kernel, dim3(blocks), dim3(threads), 0, 0, 	timeinst,
															d_initvalu,
															d_finavalu,
															d_params,
															d_com);

	hipMemcpy(finavalu, d_finavalu, d_finavalu_mem, hipMemcpyDeviceToHost);
	hipMemcpy(com, d_com, d_com_mem, hipMemcpyDeviceToHost);

	//=====================================================================
	//	FINAL KERNEL
	//=====================================================================

	initvalu_offset_ecc = 0;												// 46 points
	initvalu_offset_Dyad = 46;											// 15 points
	initvalu_offset_SL = 61;											// 15 points
	initvalu_offset_Cyt = 76;												// 15 poitns

	kernel_fin(			initvalu,
								initvalu_offset_ecc,
								initvalu_offset_Dyad,
								initvalu_offset_SL,
								initvalu_offset_Cyt,
								parameter,
								finavalu,
								com[0],
								com[1],
								com[2]);

	//=====================================================================
	//	COMPENSATION FOR NANs and INFs
	//=====================================================================

	for(i=0; i<EQUATIONS; i++){
		if (std::isnan(finavalu[i]) == 1){ 
			finavalu[i] = 0.0001;												// for NAN set rate of change to 0.0001
		}
		else if (std::isinf(finavalu[i]) == 1){ 
			finavalu[i] = 0.0001;												// for INF set rate of change to 0.0001
		}
	}

}
