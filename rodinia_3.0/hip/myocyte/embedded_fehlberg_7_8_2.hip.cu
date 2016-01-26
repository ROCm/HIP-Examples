//====================================================================================================100
//		UPDATE
//====================================================================================================100

// This file is the modified version of embedded fehlberg 7.8 solver obrained from (http://mymathlib.webtrellis.net/index.html)
// Lukasz G. Szafaryn 15 DEC 09

//====================================================================================================100
//		DESCRIPTION
//====================================================================================================100

//                                                                            //
//  Description:                                                              //
//     The Runge-Kutta-Fehlberg method is an adaptive procedure for approxi-  //
//     mating the solution of the differential equation y'(x) = f(x,y) with   //
//     initial condition y(x0) = c.  This implementation evaluates f(x,y)     //
//     thirteen times per step using embedded seventh order and eight order   //
//     Runge-Kutta estimates to estimate the not only the solution but also   //
//     the error.                                                             //
//     The next step size is then calculated using the preassigned tolerance  //
//     and error estimate.                                                    //
//     For step i+1,                                                          //
//        y[i+1] = y[i] +  h * (41/840 * k1 + 34/105 * finavalu_temp[5] + 9/35 * finavalu_temp[6]         //
//                        + 9/35 * finavalu_temp[7] + 9/280 * finavalu_temp[8] + 9/280 finavalu_temp[9] + 41/840 finavalu_temp[10] ) //
//     where                                                                  //
//     k1 = f( x[i],y[i] ),                                                   //
//     finavalu_temp[1] = f( x[i]+2h/27, y[i] + 2h*k1/27),                                  //
//     finavalu_temp[2] = f( x[i]+h/9, y[i]+h/36*( k1 + 3 finavalu_temp[1]) ),                            //
//     finavalu_temp[3] = f( x[i]+h/6, y[i]+h/24*( k1 + 3 finavalu_temp[2]) ),                            //
//     finavalu_temp[4] = f( x[i]+5h/12, y[i]+h/48*(20 k1 - 75 finavalu_temp[2] + 75 finavalu_temp[3])),                //
//     finavalu_temp[5] = f( x[i]+h/2, y[i]+h/20*( k1 + 5 finavalu_temp[3] + 4 finavalu_temp[4] ) ),                    //
//     finavalu_temp[6] = f( x[i]+5h/6, y[i]+h/108*( -25 k1 + 125 finavalu_temp[3] - 260 finavalu_temp[4] + 250 finavalu_temp[5] ) ), //
//     finavalu_temp[7] = f( x[i]+h/6, y[i]+h*( 31/300 k1 + 61/225 finavalu_temp[4] - 2/9 finavalu_temp[5]              //
//                                                            + 13/900 finavalu_temp[6]) )  //
//     finavalu_temp[8] = f( x[i]+2h/3, y[i]+h*( 2 k1 - 53/6 finavalu_temp[3] + 704/45 finavalu_temp[4] - 107/9 finavalu_temp[5]      //
//                                                      + 67/90 finavalu_temp[6] + 3 finavalu_temp[7]) ), //
//     finavalu_temp[9] = f( x[i]+h/3, y[i]+h*( -91/108 k1 + 23/108 finavalu_temp[3] - 976/135 finavalu_temp[4]        //
//                             + 311/54 finavalu_temp[5] - 19/60 finavalu_temp[6] + 17/6 finavalu_temp[7] - 1/12 finavalu_temp[8]) ), //
//     finavalu_temp[10] = f( x[i]+h, y[i]+h*( 2383/4100 k1 - 341/164 finavalu_temp[3] + 4496/1025 finavalu_temp[4]     //
//          - 301/82 finavalu_temp[5] + 2133/4100 finavalu_temp[6] + 45/82 finavalu_temp[7] + 45/164 finavalu_temp[8] + 18/41 finavalu_temp[9]) )  //
//     finavalu_temp[11] = f( x[i], y[i]+h*( 3/205 k1 - 6/41 finavalu_temp[5] - 3/205 finavalu_temp[6] - 3/41 finavalu_temp[7]        //
//                                                   + 3/41 finavalu_temp[8] + 6/41 finavalu_temp[9]) )  //
//     finavalu_temp[12] = f( x[i]+h, y[i]+h*( -1777/4100 k1 - 341/164 finavalu_temp[3] + 4496/1025 finavalu_temp[4]    //
//                      - 289/82 finavalu_temp[5] + 2193/4100 finavalu_temp[6] + 51/82 finavalu_temp[7] + 33/164 finavalu_temp[8] +   //
//                                                        12/41 finavalu_temp[9] + finavalu_temp[11]) )  //
//     x[i+1] = x[i] + h.                                                     //
//                                                                            //
//     The error is estimated to be                                           //
//        err = -41/840 * h * ( k1 + finavalu_temp[10] - finavalu_temp[11] - finavalu_temp[12])                         //
//     The step size h is then scaled by the scale factor                     //
//         scale = 0.8 * | epsilon * y[i] / [err * (xmax - x[0])] | ^ 1/7     //
//     The scale factor is further constrained 0.125 < scale < 4.0.           //
//     The new step size is h := scale * h.                                   //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//  static fp Runge_Kutta(fp (*f)(fp,fp), fp *y,          //
//                                                       fp x0, fp h) //
//                                                                            //
//  Description:                                                              //
//     This routine uses Fehlberg's embedded 7th and 8th order methods to     //
//     approximate the solution of the differential equation y'=f(x,y) with   //
//     the initial condition y = y[0] at x = x0.  The value at x + h is       //
//     returned in y[1].  The function returns err / h ( the absolute error   //
//     per step size ).                                                       //
//                                                                            //
//  Arguments:                                                                //
//     fp *f  Pointer to the function which returns the slope at (x,y) of //
//                integral curve of the differential equation y' = f(x,y)     //
//                which passes through the point (x0,y[0]).                   //
//     fp y[] On input y[0] is the initial value of y at x, on output     //
//                y[1] is the solution at x + h.                              //
//     fp x   Initial value of x.                                         //
//     fp h   Step size                                                   //
//                                                                            //
//  Return Values:                                                            //
//     This routine returns the err / h.  The solution of y(x) at x + h is    //
//     returned in y[1].                                                      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//		PARTICULAR SOLVER FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

__device__ void embedded_fehlberg_7_8_2(	fp h,

																			fp timeinst,
																			fp* initvalu,
																			fp* finavalu,
																			fp* parameter,

																			fp* error,
																			fp* initvalu_temp,
																			fp* finavalu_temp,
																			fp* com) {

	//======================================================================================================================================================
	//	VARIABLES
	//======================================================================================================================================================

	const fp c_1_11 = 41.0 / 840.0;
	const fp c6 = 34.0 / 105.0;
	const fp c_7_8= 9.0 / 35.0;
	const fp c_9_10 = 9.0 / 280.0;

	const fp a2 = 2.0 / 27.0;
	const fp a3 = 1.0 / 9.0;
	const fp a4 = 1.0 / 6.0;
	const fp a5 = 5.0 / 12.0;
	const fp a6 = 1.0 / 2.0;
	const fp a7 = 5.0 / 6.0;
	const fp a8 = 1.0 / 6.0;
	const fp a9 = 2.0 / 3.0;
	const fp a10 = 1.0 / 3.0;

	const fp b31 = 1.0 / 36.0;
	const fp b32 = 3.0 / 36.0;
	const fp b41 = 1.0 / 24.0;
	const fp b43 = 3.0 / 24.0;
	const fp b51 = 20.0 / 48.0;
	const fp b53 = -75.0 / 48.0;
	const fp b54 = 75.0 / 48.0;
	const fp b61 = 1.0 / 20.0;
	const fp b64 = 5.0 / 20.0;
	const fp b65 = 4.0 / 20.0;
	const fp b71 = -25.0 / 108.0;
	const fp b74 =  125.0 / 108.0;
	const fp b75 = -260.0 / 108.0;
	const fp b76 =  250.0 / 108.0;
	const fp b81 = 31.0/300.0;
	const fp b85 = 61.0/225.0;
	const fp b86 = -2.0/9.0;
	const fp b87 = 13.0/900.0;
	const fp b91 = 2.0;
	const fp b94 = -53.0/6.0;
	const fp b95 = 704.0 / 45.0;
	const fp b96 = -107.0 / 9.0;
	const fp b97 = 67.0 / 90.0;
	const fp b98 = 3.0;
	const fp b10_1 = -91.0 / 108.0;
	const fp b10_4 = 23.0 / 108.0;
	const fp b10_5 = -976.0 / 135.0;
	const fp b10_6 = 311.0 / 54.0;
	const fp b10_7 = -19.0 / 60.0;
	const fp b10_8 = 17.0 / 6.0;
	const fp b10_9 = -1.0 / 12.0;
	const fp b11_1 = 2383.0 / 4100.0;
	const fp b11_4 = -341.0 / 164.0;
	const fp b11_5 = 4496.0 / 1025.0;
	const fp b11_6 = -301.0 / 82.0;
	const fp b11_7 = 2133.0 / 4100.0;
	const fp b11_8 = 45.0 / 82.0;
	const fp b11_9 = 45.0 / 164.0;
	const fp b11_10 = 18.0 / 41.0;
	const fp b12_1 = 3.0 / 205.0;
	const fp b12_6 = - 6.0 / 41.0;
	const fp b12_7 = - 3.0 / 205.0;
	const fp b12_8 = - 3.0 / 41.0;
	const fp b12_9 = 3.0 / 41.0;
	const fp b12_10 = 6.0 / 41.0;
	const fp b13_1 = -1777.0 / 4100.0;
	const fp b13_4 = -341.0 / 164.0;
	const fp b13_5 = 4496.0 / 1025.0;
	const fp b13_6 = -289.0 / 82.0;
	const fp b13_7 = 2193.0 / 4100.0;
	const fp b13_8 = 51.0 / 82.0;
	const fp b13_9 = 33.0 / 164.0;
	const fp b13_10 = 12.0 / 41.0;

	const fp err_factor  = -41.0 / 840.0;

	fp h2_7 = a2 * h;

	fp timeinst_temp;

	int i,j;

	//======================================================================================================================================================
	//		EVALUATIONS
	//======================================================================================================================================================

	for(j=0; j<13; j++){

		//===================================================================================================
		//		0
		//===================================================================================================

		if(j==0){

			timeinst_temp = timeinst;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] ;
			}

		}

		//===================================================================================================
		//		1
		//===================================================================================================

		else if(j==1){

			timeinst_temp = timeinst+h2_7;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h2_7 * (finavalu_temp[0*EQUATIONS+i]);
			}

		}

		//===================================================================================================
		//		2
		//===================================================================================================

		else if(j==2){

			timeinst_temp = timeinst+a3*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b31*finavalu_temp[0*EQUATIONS+i] + b32*finavalu_temp[1*EQUATIONS+i]);
			}

		}

		//===================================================================================================
		//		3
		//===================================================================================================

		else if(j==3){

			timeinst_temp = timeinst+a4*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b41*finavalu_temp[0*EQUATIONS+i] + b43*finavalu_temp[2*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		4
		//===================================================================================================

		else if(j==4){

			timeinst_temp = timeinst+a5*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b51*finavalu_temp[0*EQUATIONS+i] + b53*finavalu_temp[2*EQUATIONS+i] + b54*finavalu_temp[3*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		5
		//===================================================================================================

		else if(j==5){

			timeinst_temp = timeinst+a6*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b61*finavalu_temp[0*EQUATIONS+i] + b64*finavalu_temp[3*EQUATIONS+i] + b65*finavalu_temp[4*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		6
		//===================================================================================================

		else if(j==6){

			timeinst_temp = timeinst+a7*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b71*finavalu_temp[0*EQUATIONS+i] + b74*finavalu_temp[3*EQUATIONS+i] + b75*finavalu_temp[4*EQUATIONS+i] + b76*finavalu_temp[5*EQUATIONS+i]);
			}

		}

		//===================================================================================================
		//		7
		//===================================================================================================

		else if(j==7){

			timeinst_temp = timeinst+a8*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b81*finavalu_temp[0*EQUATIONS+i] + b85*finavalu_temp[4*EQUATIONS+i] + b86*finavalu_temp[5*EQUATIONS+i] + b87*finavalu_temp[6*EQUATIONS+i]);
			}

		}

		//===================================================================================================
		//		8
		//===================================================================================================

		else if(j==8){

			timeinst_temp = timeinst+a9*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b91*finavalu_temp[0*EQUATIONS+i] + b94*finavalu_temp[3*EQUATIONS+i] + b95*finavalu_temp[4*EQUATIONS+i] + b96*finavalu_temp[5*EQUATIONS+i] + b97*finavalu_temp[6*EQUATIONS+i]+ b98*finavalu_temp[7*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		9
		//===================================================================================================

		else if(j==9){

			timeinst_temp = timeinst+a10*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b10_1*finavalu_temp[0*EQUATIONS+i] + b10_4*finavalu_temp[3*EQUATIONS+i] + b10_5*finavalu_temp[4*EQUATIONS+i] + b10_6*finavalu_temp[5*EQUATIONS+i] + b10_7*finavalu_temp[6*EQUATIONS+i] + b10_8*finavalu_temp[7*EQUATIONS+i] + b10_9*finavalu_temp[8*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		10
		//===================================================================================================

		else if(j==10){

			timeinst_temp = timeinst+h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b11_1*finavalu_temp[0*EQUATIONS+i] + b11_4*finavalu_temp[3*EQUATIONS+i] + b11_5*finavalu_temp[4*EQUATIONS+i] + b11_6*finavalu_temp[5*EQUATIONS+i] + b11_7*finavalu_temp[6*EQUATIONS+i] + b11_8*finavalu_temp[7*EQUATIONS+i] + b11_9*finavalu_temp[8*EQUATIONS+i]+ b11_10 * finavalu_temp[9*EQUATIONS+i]);
			}

		}

		//===================================================================================================
		//		11
		//===================================================================================================

		else if(j==11){

			timeinst_temp = timeinst;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b12_1*finavalu_temp[0*EQUATIONS+i] + b12_6*finavalu_temp[5*EQUATIONS+i] + b12_7*finavalu_temp[6*EQUATIONS+i] + b12_8*finavalu_temp[7*EQUATIONS+i] + b12_9*finavalu_temp[8*EQUATIONS+i] + b12_10 * finavalu_temp[9*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		12
		//===================================================================================================

		else if(j==12){

			timeinst_temp = timeinst+h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b13_1*finavalu_temp[0*EQUATIONS+i] + b13_4*finavalu_temp[3*EQUATIONS+i] + b13_5*finavalu_temp[4*EQUATIONS+i] + b13_6*finavalu_temp[5*EQUATIONS+i] + b13_7*finavalu_temp[6*EQUATIONS+i] + b13_8*finavalu_temp[7*EQUATIONS+i] + b13_9*finavalu_temp[8*EQUATIONS+i] + b13_10*finavalu_temp[9*EQUATIONS+i] + finavalu_temp[11*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		EVALUATION
		//===================================================================================================

		kernel_2(	timeinst_temp,
							initvalu_temp,
							parameter,
							&finavalu_temp[j*EQUATIONS],
							com);

	}

	//======================================================================================================================================================
	//		FINAL VALUE
	//======================================================================================================================================================

	for(i=0; i<EQUATIONS; i++){
		finavalu[i]= initvalu[i] +  h * (c_1_11 * (finavalu_temp[0*EQUATIONS+i] + finavalu_temp[10*EQUATIONS+i])  + c6 * finavalu_temp[5*EQUATIONS+i] + c_7_8 * (finavalu_temp[6*EQUATIONS+i] + finavalu_temp[7*EQUATIONS+i]) + c_9_10 * (finavalu_temp[8*EQUATIONS+i] + finavalu_temp[9*EQUATIONS+i]) );
		// printf("finavalu_temp[0][%d] = %f\n", i, finavalu_temp[0][i]);
		// printf("finavalu_temp[10][%d] = %f\n", i, finavalu_temp[10][i]);
		// printf("finavalu_temp[5][%d] = %f\n", i, finavalu_temp[5][i]);
		// printf("finavalu_temp[6][%d] = %f\n", i, finavalu_temp[6][i]);
		// printf("finavalu_temp[7][%d] = %f\n", i, finavalu_temp[7][i]);
		// printf("finavalu_temp[8][%d] = %f\n", i, finavalu_temp[8][i]);
		// printf("finavalu_temp[9][%d] = %f\n", i, finavalu_temp[9][i]);
		// printf("finavalu[%d] = %f\n", i, finavalu[i]);
	}

	//======================================================================================================================================================
	//		RETURN
	//======================================================================================================================================================

	for(i=0; i<EQUATIONS; i++){
		error[i] = fabs(err_factor * (finavalu_temp[0*EQUATIONS+i] + finavalu_temp[10*EQUATIONS+i] - finavalu_temp[11*EQUATIONS+i] - finavalu_temp[12*EQUATIONS+i]));
		// printf("Error[%d] = %f\n", i, error[i]);
	}

}
