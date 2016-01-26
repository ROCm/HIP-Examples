//=====================================================================
//	MAIN FUNCTION
//=====================================================================
__device__ void kernel_fin_2(	int timeinst,
													fp* d_initvalu,
													fp* d_finavalu,
													int offset_ecc,
													int offset_Dyad,
													int offset_SL,
													int offset_Cyt,
													fp* d_params,
													fp* d_com){

//=====================================================================
//	VARIABLES
//=====================================================================

	// input parameters
	fp BtotDyad;
	fp CaMKIItotDyad;

	// compute variables
	fp Vmyo;																			// [L]
	fp Vdyad;																			// [L]
	fp VSL;																				// [L]
	// fp kDyadSL;																			// [L/msec]
	fp kSLmyo;																			// [L/msec]
	fp k0Boff;																			// [s^-1] 
	fp k0Bon;																			// [uM^-1 s^-1] kon = koff/Kd
	fp k2Boff;																			// [s^-1] 
	fp k2Bon;																			// [uM^-1 s^-1]
	// fp k4Boff;																			// [s^-1]
	fp k4Bon;																			// [uM^-1 s^-1]
	fp CaMtotDyad;
	fp Bdyad;																			// [uM dyad]
	fp J_cam_dyadSL;																	// [uM/msec dyad]
	fp J_ca2cam_dyadSL;																	// [uM/msec dyad]
	fp J_ca4cam_dyadSL;																	// [uM/msec dyad]
	fp J_cam_SLmyo;																		// [umol/msec]
	fp J_ca2cam_SLmyo;																	// [umol/msec]
	fp J_ca4cam_SLmyo;																	// [umol/msec]

//=====================================================================
//	COMPUTATION
//=====================================================================

	// input parameters
	BtotDyad = d_params[1];
	CaMKIItotDyad = d_params[2];

	// ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec to mM/msec
	d_finavalu[offset_ecc+35] = d_finavalu[offset_ecc+35] + 1e-3*d_com[0];
	d_finavalu[offset_ecc+36] = d_finavalu[offset_ecc+36] + 1e-3*d_com[1];
	d_finavalu[offset_ecc+37] = d_finavalu[offset_ecc+37] + 1e-3*d_com[2]; 

	// incorporate CaM diffusion between compartments
	Vmyo = 2.1454e-11;																// [L]
	Vdyad = 1.7790e-14;																// [L]
	VSL = 6.6013e-13;																// [L]
	// kDyadSL = 3.6363e-16;															// [L/msec]
	kSLmyo = 8.587e-15;																// [L/msec]
	k0Boff = 0.0014;																// [s^-1] 
	k0Bon = k0Boff/0.2;																// [uM^-1 s^-1] kon = koff/Kd
	k2Boff = k0Boff/100;															// [s^-1] 
	k2Bon = k0Bon;																	// [uM^-1 s^-1]
	// k4Boff = k2Boff;																// [s^-1]
	k4Bon = k0Bon;																	// [uM^-1 s^-1]
	CaMtotDyad = d_initvalu[offset_Dyad+0]
			   + d_initvalu[offset_Dyad+1]
			   + d_initvalu[offset_Dyad+2]
			   + d_initvalu[offset_Dyad+3]
			   + d_initvalu[offset_Dyad+4]
			   + d_initvalu[offset_Dyad+5]
			   + CaMKIItotDyad * (	  d_initvalu[offset_Dyad+6]
												  + d_initvalu[offset_Dyad+7]
												  + d_initvalu[offset_Dyad+8]
												  + d_initvalu[offset_Dyad+9])
			   + d_initvalu[offset_Dyad+12]
			   + d_initvalu[offset_Dyad+13]
			   + d_initvalu[offset_Dyad+14];
	Bdyad = BtotDyad - CaMtotDyad;																				// [uM dyad]
	J_cam_dyadSL = 1e-3 * (  k0Boff*d_initvalu[offset_Dyad+0] - k0Bon*Bdyad*d_initvalu[offset_SL+0]);			// [uM/msec dyad]
	J_ca2cam_dyadSL = 1e-3 * (  k2Boff*d_initvalu[offset_Dyad+1] - k2Bon*Bdyad*d_initvalu[offset_SL+1]);		// [uM/msec dyad]
	J_ca4cam_dyadSL = 1e-3 * (  k2Boff*d_initvalu[offset_Dyad+2] - k4Bon*Bdyad*d_initvalu[offset_SL+2]);		// [uM/msec dyad]

	J_cam_SLmyo = kSLmyo * (  d_initvalu[offset_SL+0] - d_initvalu[offset_Cyt+0]);								// [umol/msec]
	J_ca2cam_SLmyo = kSLmyo * (  d_initvalu[offset_SL+1] - d_initvalu[offset_Cyt+1]);							// [umol/msec]
	J_ca4cam_SLmyo = kSLmyo * (  d_initvalu[offset_SL+2] - d_initvalu[offset_Cyt+2]);							// [umol/msec]

	// ADJUST CAM Dyad 
	d_finavalu[offset_Dyad+0] = d_finavalu[offset_Dyad+0] - J_cam_dyadSL;
	d_finavalu[offset_Dyad+1] = d_finavalu[offset_Dyad+1] - J_ca2cam_dyadSL;
	d_finavalu[offset_Dyad+2] = d_finavalu[offset_Dyad+2] - J_ca4cam_dyadSL;

	// ADJUST CAM Sl
	d_finavalu[offset_SL+0] = d_finavalu[offset_SL+0] + J_cam_dyadSL*Vdyad/VSL - J_cam_SLmyo/VSL;
	d_finavalu[offset_SL+1] = d_finavalu[offset_SL+1] + J_ca2cam_dyadSL*Vdyad/VSL - J_ca2cam_SLmyo/VSL;
	d_finavalu[offset_SL+2] = d_finavalu[offset_SL+2] + J_ca4cam_dyadSL*Vdyad/VSL - J_ca4cam_SLmyo/VSL;

	// ADJUST CAM Cyt 
	d_finavalu[offset_Cyt+0] = d_finavalu[offset_Cyt+0] + J_cam_SLmyo/Vmyo;
	d_finavalu[offset_Cyt+1] = d_finavalu[offset_Cyt+1] + J_ca2cam_SLmyo/Vmyo;
	d_finavalu[offset_Cyt+2] = d_finavalu[offset_Cyt+2] + J_ca4cam_SLmyo/Vmyo;

}
