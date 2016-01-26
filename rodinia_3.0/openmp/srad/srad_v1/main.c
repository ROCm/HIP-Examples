//====================================================================================================100
//		UPDATE
//====================================================================================================100

//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a  
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts   
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments

//====================================================================================================100
//	DEFINE / INCLUDE
//====================================================================================================100

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "define.c"
#include "graphics.c"
#include "resize.c"
#include "timer.c"

//====================================================================================================100
//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100
//====================================================================================================100

int main(int argc, char *argv []){

	//================================================================================80
	// 	VARIABLES
	//================================================================================80

	// time
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;
	long long time7;
	long long time8;
	long long time9;
	long long time10;

	time0 = get_time();

    // inputs image, input paramenters
    fp* image_ori;																// originalinput image
	int image_ori_rows;
	int image_ori_cols;
	long image_ori_elem;

    // inputs image, input paramenters
    fp* image;															// input image
    long Nr,Nc;													// IMAGE nbr of rows/cols/elements
	long Ne;

	// algorithm parameters
    int niter;																// nbr of iterations
    fp lambda;															// update step size

    // size of IMAGE
	int r1,r2,c1,c2;												// row/col coordinates of uniform ROI
	long NeROI;														// ROI nbr of elements
    
    // ROI statistics
    fp meanROI, varROI, q0sqr;											//local region statistics
    
    // surrounding pixel indicies
    int *iN,*iS,*jE,*jW;    

    // center pixel value
    fp Jc;

	// directional derivatives
	fp *dN,*dS,*dW,*dE;
    
    // calculation variables
    fp tmp,sum,sum2;
    fp G2,L,num,den,qsqr,D;
       
    // diffusion coefficient
    fp *c; 
	fp cN,cS,cW,cE;
    
    // counters
    int iter;   // primary loop
    long i,j;    // image row/col
    long k;      // image single index    

	// number of threads
	int threads;

	time1 = get_time();

	//================================================================================80
	// 	GET INPUT PARAMETERS
	//================================================================================80

	if(argc != 6){
		printf("ERROR: wrong number of arguments\n");
		return 0;
	}
	else{
		niter = atoi(argv[1]);
		lambda = atof(argv[2]);
		Nr = atoi(argv[3]);						// it is 502 in the original image
		Nc = atoi(argv[4]);						// it is 458 in the original image
		threads = atoi(argv[5]);
	}

	omp_set_num_threads(threads);
	// printf("THREAD %d\n", omp_get_thread_num());
	// printf("NUMBER OF THREADS: %d\n", omp_get_num_threads());

	time2 = get_time();

	//================================================================================80
	// 	READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	//================================================================================80

    // read image
	image_ori_rows = 502;
	image_ori_cols = 458;
	image_ori_elem = image_ori_rows * image_ori_cols;

	image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

	read_graphics(	"../../../data/srad/image.pgm",
								image_ori,
								image_ori_rows,
								image_ori_cols,
								1);

	time3 = get_time();

	//================================================================================80
	// 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	//================================================================================80

	Ne = Nr*Nc;

	image = (fp*)malloc(sizeof(fp) * Ne);

	resize(	image_ori,
				image_ori_rows,
				image_ori_cols,
				image,
				Nr,
				Nc,
				1);

	time4 = get_time();

	//================================================================================80
	// 	SETUP
	//================================================================================80

    r1     = 0;											// top row index of ROI
    r2     = Nr - 1;									// bottom row index of ROI
    c1     = 0;											// left column index of ROI
    c2     = Nc - 1;									// right column index of ROI

    // ROI image size    
    NeROI = (r2-r1+1)*(c2-c1+1);											// number of elements in ROI, ROI size
    
    // allocate variables for surrounding pixels
    iN = malloc(sizeof(int*)*Nr) ;									// north surrounding element
    iS = malloc(sizeof(int*)*Nr) ;									// south surrounding element
    jW = malloc(sizeof(int*)*Nc) ;									// west surrounding element
    jE = malloc(sizeof(int*)*Nc) ;									// east surrounding element
    
	// allocate variables for directional derivatives
	dN = malloc(sizeof(fp)*Ne) ;											// north direction derivative
    dS = malloc(sizeof(fp)*Ne) ;											// south direction derivative
    dW = malloc(sizeof(fp)*Ne) ;											// west direction derivative
    dE = malloc(sizeof(fp)*Ne) ;											// east direction derivative

	// allocate variable for diffusion coefficient
    c  = malloc(sizeof(fp)*Ne) ;											// diffusion coefficient
        
    // N/S/W/E indices of surrounding pixels (every element of IMAGE)
	// #pragma omp parallel
    for (i=0; i<Nr; i++) {
        iN[i] = i-1;														// holds index of IMAGE row above
        iS[i] = i+1;														// holds index of IMAGE row below
    }
	// #pragma omp parallel
    for (j=0; j<Nc; j++) {
        jW[j] = j-1;														// holds index of IMAGE column on the left
        jE[j] = j+1;														// holds index of IMAGE column on the right
    }
	// N/S/W/E boundary conditions, fix surrounding indices outside boundary of IMAGE
    iN[0]    = 0;															// changes IMAGE top row index from -1 to 0
    iS[Nr-1] = Nr-1;														// changes IMAGE bottom row index from Nr to Nr-1 
    jW[0]    = 0;															// changes IMAGE leftmost column index from -1 to 0
    jE[Nc-1] = Nc-1;														// changes IMAGE rightmost column index from Nc to Nc-1

	time5 = get_time();

	//================================================================================80
	// 	SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//================================================================================80

	// #pragma omp parallel
	for (i=0; i<Ne; i++) {													// do for the number of elements in input IMAGE
		image[i] = exp(image[i]/255);											// exponentiate input IMAGE and copy to output image
    }

	time6 = get_time();

	//================================================================================80
	// 	COMPUTATION
	//================================================================================80

	// printf("iterations: ");

    // primary loop
    for (iter=0; iter<niter; iter++){										// do for the number of iterations input parameter

		// printf("%d ", iter);
		// fflush(NULL);

        // ROI statistics for entire ROI (single number for ROI)
        sum=0; 
		sum2=0;
        for (i=r1; i<=r2; i++) {											// do for the range of rows in ROI
            for (j=c1; j<=c2; j++) {										// do for the range of columns in ROI
                tmp   = image[i + Nr*j];										// get coresponding value in IMAGE
                sum  += tmp ;												// take corresponding value and add to sum
                sum2 += tmp*tmp;											// take square of corresponding value and add to sum2
            }
        }
        meanROI = sum / NeROI;												// gets mean (average) value of element in ROI
        varROI  = (sum2 / NeROI) - meanROI*meanROI;							// gets variance of ROI
        q0sqr   = varROI / (meanROI*meanROI);								// gets standard deviation of ROI

        // directional derivatives, ICOV, diffusion coefficent
		#pragma omp parallel for shared(image, dN, dS, dW, dE, c, Nr, Nc, iN, iS, jW, jE) private(i, j, k, Jc, G2, L, num, den, qsqr)
		for (j=0; j<Nc; j++) {												// do for the range of columns in IMAGE

            for (i=0; i<Nr; i++) {											// do for the range of rows in IMAGE 

                // current index/pixel
                k = i + Nr*j;												// get position of current element
                Jc = image[k];													// get value of the current element

                // directional derivates (every element of IMAGE)
                dN[k] = image[iN[i] + Nr*j] - Jc;								// north direction derivative
                dS[k] = image[iS[i] + Nr*j] - Jc;								// south direction derivative
                dW[k] = image[i + Nr*jW[j]] - Jc;								// west direction derivative
                dE[k] = image[i + Nr*jE[j]] - Jc;								// east direction derivative

                // normalized discrete gradient mag squared (equ 52,53)
                G2 = (dN[k]*dN[k] + dS[k]*dS[k]								// gradient (based on derivatives)
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

                // normalized discrete laplacian (equ 54)
                L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;					// laplacian (based on derivatives)

                // ICOV (equ 31/35)
                num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;						// num (based on gradient and laplacian)
                den  = 1 + (.25*L);											// den (based on laplacian)
                qsqr = num/(den*den);										// qsqr (based on num and den)
 
                // diffusion coefficent (equ 33) (every element of IMAGE)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;					// den (based on qsqr and q0sqr)
                c[k] = 1.0 / (1.0+den) ;									// diffusion coefficient (based on den)

                // saturate diffusion coefficent to 0-1 range
                if (c[k] < 0)												// if diffusion coefficient < 0
					{c[k] = 0;}												// ... set to 0
                else if (c[k] > 1)											// if diffusion coefficient > 1
					{c[k] = 1;}												// ... set to 1

            }

        }

        // divergence & image update
		#pragma omp parallel for shared(image, c, Nr, Nc, lambda) private(i, j, k, D, cS, cN, cW, cE)
        for (j=0; j<Nc; j++) {												// do for the range of columns in IMAGE

			// printf("NUMBER OF THREADS: %d\n", omp_get_num_threads());

            for (i=0; i<Nr; i++) {											// do for the range of rows in IMAGE

                // current index
                k = i + Nr*j;												// get position of current element

                // diffusion coefficent
                cN = c[k];													// north diffusion coefficient
                cS = c[iS[i] + Nr*j];										// south diffusion coefficient
                cW = c[k];													// west diffusion coefficient
                cE = c[i + Nr*jE[j]];										// east diffusion coefficient

                // divergence (equ 58)
                D = cN*dN[k] + cS*dS[k] + cW*dW[k] + cE*dE[k];				// divergence

                // image update (equ 61) (every element of IMAGE)
                image[k] = image[k] + 0.25*lambda*D;								// updates image (based on input time step and divergence)

            }

        }

	}

	// printf("\n");

	time7 = get_time();

	//================================================================================80
	// 	SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//================================================================================80

	// #pragma omp parallel
	for (i=0; i<Ne; i++) {													// do for the number of elements in IMAGE
		image[i] = log(image[i])*255;													// take logarithm of image, log compress
	}

	time8 = get_time();

	//================================================================================80
	// 	WRITE IMAGE AFTER PROCESSING
	//================================================================================80

	write_graphics(	"image_out.pgm",
								image,
								Nr,
								Nc,
								1,
								255);

	time9 = get_time();

	//================================================================================80
	// 	DEALLOCATE
	//================================================================================80

	free(image_ori);
	free(image);

    free(iN); free(iS); free(jW); free(jE);									// deallocate surrounding pixel memory
    free(dN); free(dS); free(dW); free(dE);									// deallocate directional derivative memory
    free(c);																// deallocate diffusion coefficient memory

	time10 = get_time();

	//================================================================================80
	//		DISPLAY TIMING
	//================================================================================80

	printf("Time spent in different stages of the application:\n");
	printf("%.12f s, %.12f % : SETUP VARIABLES\n", 									(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : READ COMMAND LINE PARAMETERS\n", 	(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : READ IMAGE FROM FILE\n", 						(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : RESIZE IMAGE\n", 										(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : SETUP, MEMORY ALLOCATION\n", 				(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : EXTRACT IMAGE\n", 									(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : COMPUTE\n", 												(float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : COMPRESS IMAGE\n", 									(float) (time8-time7) / 1000000, (float) (time8-time7) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : SAVE IMAGE INTO FILE\n", 							(float) (time9-time8) / 1000000, (float) (time9-time8) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : FREE MEMORY\n", 										(float) (time10-time9) / 1000000, (float) (time10-time9) / (float) (time10-time0) * 100);
	printf("Total time:\n");
	printf("%.12f s\n", 																					(float) (time10-time0) / 1000000);

//====================================================================================================100
//	END OF FILE
//====================================================================================================100

}


