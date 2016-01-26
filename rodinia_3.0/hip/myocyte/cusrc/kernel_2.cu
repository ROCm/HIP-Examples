//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//		KERNEL
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

__device__ void kernel_2(	int timeinst,
											fp* initvalu,
											fp* params,
											fp* finavalu,
											fp* com){

	//======================================================================================================================================================
	// 	VARIABLES
	//======================================================================================================================================================

	// pointers
	int valu_offset_ecc;															// inivalu and finavalu offset
	int valu_offset_Dyad;														// Dyad value offset
	int valu_offset_SL;																// SL value offset
	int valu_offset_Cyt;															// Cyt value offset

	int params_offset_Dyad;													// Dyad parameters offset
	int params_offset_SL;														// SL parameters offset
	int params_offset_Cyt;														// Cyt parameters offset

	int com_offset_Dyad;															// kernel1-kernel2 Dyad communication offset
	int com_offset_SL;																// kernel1-kernel2 SL communication offset
	int com_offset_Cyt;															// kernel-kernel Cyt communication offset

	// module parameters
	fp CaDyad;																			// from ECC model, *** Converting from [mM] to [uM] ***
	fp CaSL;																				// from ECC model, *** Converting from [mM] to [uM] ***
	fp CaCyt;																			// from ECC model, *** Converting from [mM] to [uM] ***

	// counter
	int i;

	//======================================================================================================================================================
	// 	COMPUTATION
	//======================================================================================================================================================

	valu_offset_ecc = 0;
	valu_offset_Dyad = 46;
	valu_offset_SL = 61;
	valu_offset_Cyt = 76;

	params_offset_Dyad = 0;
	params_offset_SL = 5;
	params_offset_Cyt = 10;

	com_offset_Dyad = 0;
	com_offset_SL = 1;
	com_offset_Cyt = 2;

	//==================================================
	//		ECC
	//==================================================

	// ecc function
	kernel_ecc_2(	timeinst,
							initvalu,
							finavalu,
							valu_offset_ecc,
							params);

	//==================================================
	//		3xCAM
	//==================================================

	// specific
	CaDyad = initvalu[35]*1e3;							// from ECC model, *** Converting from [mM] to [uM] ***
	// cam function for Dyad
	kernel_cam_2(	timeinst,
								initvalu,
								finavalu,
								valu_offset_Dyad,
								params,
								params_offset_Dyad,
								com,
								com_offset_Dyad,
								CaDyad);

	// specific
	CaSL = initvalu[36]*1e3;								// from ECC model, *** Converting from [mM] to [uM] ***
	// cam function for Dyad
	kernel_cam_2(	timeinst,
								initvalu,
								finavalu,
								valu_offset_SL,
								params,
								params_offset_SL,
								com,
								com_offset_SL,
								CaSL);

	// specific
	CaCyt = initvalu[37]*1e3;							// from ECC model, *** Converting from [mM] to [uM] ***
	// cam function for Dyad
	kernel_cam_2(	timeinst,
								initvalu,
								finavalu,
								valu_offset_Cyt,
								params,
								params_offset_Cyt,
								com,
								com_offset_Cyt,
								CaCyt);

	//====================================================================================================
	//		SEGMENT HAPPENING 2ND IN TIME: FINAL
	//====================================================================================================

	kernel_fin_2(	timeinst,
							initvalu,
							finavalu,
							valu_offset_ecc,
							valu_offset_Dyad,
							valu_offset_SL,
							valu_offset_Cyt,
							params,
							com);

	//====================================================================================================
	//		make sure function does not return NANs and INFs
	//====================================================================================================

	for(i=0; i<EQUATIONS; i++){
		if (isnan(finavalu[i]) == 1){ 
			finavalu[i] = 0.0001;												// for NAN set rate of change to 0.0001
		}
		else if (isinf(finavalu[i]) == 1){ 
			finavalu[i] = 0.0001;												// for INF set rate of change to 0.0001
		}
	}

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//		END OF KERNEL
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

}
