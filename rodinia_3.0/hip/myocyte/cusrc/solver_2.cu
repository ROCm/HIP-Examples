//======================================================================================================================================================
//======================================================================================================================================================
//		INCLUDE
//======================================================================================================================================================
//======================================================================================================================================================

#include <math.h>

#define max(x,y) ( (x) < (y) ? (y) : (x) )
#define min(x,y) ( (x) < (y) ? (x) : (y) )

#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.0

//======================================================================================================================================================
//======================================================================================================================================================
//		SOLVER FUNCTION
//======================================================================================================================================================
//======================================================================================================================================================

__global__ void solver_2(	int workload,
											int xmax,

											fp* x,
											fp* y,
											fp* params,

											fp* com,
											fp* err,
											fp* scale,
											fp* yy,
											fp* initvalu_temp,
											fp* finavalu_temp){

	//========================================================================================================================
	//	VARIABLES
	//========================================================================================================================

	// CUDA indexes
	int bx;																					// get current horizontal block index (0-n)
	int tx;																					// get current horizontal thread index (0-n)
	int tid;																					// thread identifier

	// pointers
	long y_pointer_initial;
	long y_pointer_current; 
	long x_pointer_current;
	int err_pointer;
	int scale_pointer;
	int yy_pointer;
	int params_pointer;
	int initvalu_temp_pointer;
	int finavalu_temp_pointer;
	int com_pointer;

	// solver parameters
	fp err_exponent ;
	fp h_init;
	fp h;
	fp tolerance;
	int xmin;

	// temporary solver variables
	int error;
	int outside;
	fp scale_min;
	fp scale_fina;

	// counters
	int i, j, k;

	//========================================================================================================================
	//		INITIAL SETUP
	//========================================================================================================================

	// CUDA indexes
	bx = blockIdx.x;																// get current horizontal block index (0-n)
	tx = threadIdx.x;																// get current horizontal thread index (0-n)
	tid = bx*NUMBER_THREADS+tx;

	// save pointers, these pointers are one per workload, independent of time step
	err_pointer = tid*EQUATIONS;
	scale_pointer = tid*EQUATIONS;
	yy_pointer = tid*EQUATIONS;
	params_pointer = tid*PARAMETERS;
	initvalu_temp_pointer = tid*EQUATIONS;
	finavalu_temp_pointer = tid*13*EQUATIONS;
	com_pointer = tid*3;

	// solver parameters
	err_exponent = 1.0 / 7.0;
	h_init = 1;
	h = h_init;
	xmin = 0;
	tolerance = 10 / (fp)(xmax - xmin);

	//========================================================================================================================
	//		RANGE AND STEP CHECKING
	//========================================================================================================================

	// Verify that the step size is positive and that the upper endpoint of integration is greater than the initial enpoint.               //
	if (xmax < xmin || h <= 0.0){
		return;
	}

	// If the upper endpoint of the independent variable agrees with the initial value of the independent variable.  Set the value of the dependent variable and return success. //
	if (xmax == xmin){
		return; 
	}

	// Insure that the step size h is not larger than the length of the integration interval.                                            //
	if (h > (xmax - xmin) ) { 
		h = (fp)xmax - (fp)xmin; 
	}

	//========================================================================================================================
	//		SOLVING IF THERE ARE MANY WORKLOADS
	//========================================================================================================================

	// limit to useful threads
	if(tid<workload){

		for(k=1; k<(xmax+1); k++) {											// start after initial value

			y_pointer_initial = tid*((xmax+1)*EQUATIONS)+(k-1)*EQUATIONS;
			y_pointer_current = tid*((xmax+1)*EQUATIONS)+k*EQUATIONS;
			x_pointer_current = tid*(xmax+1)+k;

			x[x_pointer_current] = (fp)k;															// set this to k if you want time incremente with respect to previous to be k+h, set this to k-1 if you want the increment to be h
			h = h_init;

			//==========================================================================================
			//		REINITIALIZE VARIABLES
			//==========================================================================================

			scale_fina = 1.0;

			//==========================================================================================
			//		MAKE ATTEMPTS TO MINIMIZE ERROR
			//==========================================================================================

			// make attempts to minimize error
			for (j = 0; j < ATTEMPTS; j++) {

				//============================================================
				//		REINITIALIZE VARIABLES
				//============================================================

				error = 0;
				outside = 0;
				scale_min = MAX_SCALE_FACTOR;

				//============================================================
				//		EVALUATE ALL EQUATIONS
				//============================================================

				embedded_fehlberg_7_8_2(	h,																												// single value

																x[x_pointer_current],																				// single value
																&y[y_pointer_initial],																					// 91 array
																&y[y_pointer_current],																				// 91 array
																&params[params_pointer],																		// 18 array

																&err[err_pointer],																						// 91 array
																&initvalu_temp[initvalu_temp_pointer],													// 91 array
																&finavalu_temp[finavalu_temp_pointer],													// 13*91 array
																&com[com_pointer]);																				// 3 array

				//============================================================
				//		IF THERE WAS NO ERROR FOR ANY OF EQUATIONS, SET SCALE AND LEAVE THE LOOP
				//============================================================

				for(i=0; i<EQUATIONS; i++){
					if(err[err_pointer+i] > 0){
						error = 1;
					}
				}
				if (error != 1) {
					scale_fina = MAX_SCALE_FACTOR; 
					break;
				}

				//============================================================
				//		FIGURE OUT SCALE AS THE MINIMUM OF COMPONENT SCALES
				//============================================================

				for(i=0; i<EQUATIONS; i++){
					if(y[y_pointer_initial+i] == 0.0){
						yy[yy_pointer+i] = tolerance;
					}
					else{
						yy[yy_pointer+i] = fabs(y[y_pointer_initial+i]);
					}
					scale[scale_pointer+i] = 0.8 * pow( tolerance * yy[yy_pointer+i] / err[err_pointer+i] , err_exponent );
					if(scale[scale_pointer+i]<scale_min){
						scale_min = scale[scale_pointer+i];
					}
				}
				scale_fina = min( max(scale_min,MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);

				//============================================================
				//		IF WITHIN TOLERANCE, FINISH ATTEMPTS...
				//============================================================

				for(i=0; i<EQUATIONS; i++){
					if ( err[err_pointer+i] > ( tolerance * yy[yy_pointer+i] ) ){
						outside = 1;
					}
				}
				if (outside == 0){
					break;
				}

				//============================================================
				//		...OTHERWISE, ADJUST STEP FOR NEXT ATTEMPT
				//============================================================

				// scale next step in a default way
				h = h * scale_fina;

				// limit step to 0.9, because when it gets close to 1, it no longer makes sense, as 1 is already the next time instance (added to original algorithm)
				if (h >= 0.9) {
					h = 0.9;
				}

				// if instance+step exceeds range limit, limit to that range
				if ( x[x_pointer_current] + h > (fp)xmax ){
					h = (fp)xmax - x[x_pointer_current];
				}

				// if getting closer to range limit, decrease step
				else if ( x[x_pointer_current] + h + 0.5 * h > (fp)xmax ){
					h = 0.5 * h;
				}

			}

			//==========================================================================================
			//		SAVE TIME INSTANCE THAT SOLVER ENDED UP USING
			//==========================================================================================

			x[x_pointer_current] = x[x_pointer_current] + h;

			//==========================================================================================
			//		IF MAXIMUM NUMBER OF ATTEMPTS REACHED AND CANNOT GIVE SOLUTION, EXIT PROGRAM WITH ERROR
			//==========================================================================================

			if ( j >= ATTEMPTS ) {
				return; 
			}

		}

	}

	//========================================================================================================================
	//		FINAL RETURN
	//========================================================================================================================

	return;

//======================================================================================================================================================
//======================================================================================================================================================
//		END OF SOLVER FUNCTION
//======================================================================================================================================================
//======================================================================================================================================================

} 
