//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//		KERNEL
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

__global__ void kernel(	int timeinst,
										fp* d_initvalu,
										fp* d_finavalu,
										fp* d_params,
										fp* d_com){

	//======================================================================================================================================================
	// 	VARIABLES
	//======================================================================================================================================================

	// CUDA indexes
	int bx;																					// get current horizontal block index (0-n)
	int tx;																					// get current horizontal thread index (0-n)

	// pointers
	int valu_offset;																	// inivalu and finavalu offset
	int params_offset;																// parameters offset
	int com_offset;																	// kernel1-kernel2 communication offset

	// module parameters
	fp CaDyad;																			// from ECC model, *** Converting from [mM] to [uM] ***
	fp CaSL;																				// from ECC model, *** Converting from [mM] to [uM] ***
	fp CaCyt;																			// from ECC model, *** Converting from [mM] to [uM] ***

	//======================================================================================================================================================
	// 	COMPUTATION
	//======================================================================================================================================================

	// CUDA indexes
	bx = blockIdx.x;																// get current horizontal block index (0-n)
	tx = threadIdx.x;																// get current horizontal thread index (0-n)

	//=====================================================================
	//		ECC
	//=====================================================================

	// limit to useful threads
	if(bx == 0){																		// first processor runs ECC
		if(tx == 0){																	// only 1 thread runs it, since its a sequential code

			// thread offset
			valu_offset = 0;															//
			// ecc function
			kernel_ecc(	timeinst,
								d_initvalu,
								d_finavalu,
								valu_offset,
								d_params);

		}
	}

	//=====================================================================
	//		CAM x 3
	//=====================================================================

	// limit to useful threads
	else if(bx == 1){																// second processor runs CAMs (in parallel with ECC)
		if(tx == 0){																	// only 1 thread runs it, since its a sequential code

			// specific
			valu_offset = 46;
			params_offset = 0;
			com_offset = 0;
			CaDyad = d_initvalu[35]*1e3;									// from ECC model, *** Converting from [mM] to [uM] ***
			// cam function for Dyad
			kernel_cam(	timeinst,
									d_initvalu,
									d_finavalu,
									valu_offset,
									d_params,
									params_offset,
									d_com,
									com_offset,
									CaDyad);

			// specific
			valu_offset = 61;
			params_offset = 5;
			com_offset = 1;
			CaSL = d_initvalu[36]*1e3;										// from ECC model, *** Converting from [mM] to [uM] ***
			// cam function for Dyad
			kernel_cam(	timeinst,
									d_initvalu,
									d_finavalu,
									valu_offset,
									d_params,
									params_offset,
									d_com,
									com_offset,
									CaSL);

			// specific
			valu_offset = 76;
			params_offset = 10;
			com_offset = 2;
			CaCyt = d_initvalu[37]*1e3;										// from ECC model, *** Converting from [mM] to [uM] ***
			// cam function for Dyad
			kernel_cam(	timeinst,
									d_initvalu,
									d_finavalu,
									valu_offset,
									d_params,
									params_offset,
									d_com,
									com_offset,
									CaCyt);

		}
	}

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//		END OF KERNEL
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

}
