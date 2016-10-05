#include "hip/hip_runtime.h"
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

//======================================================================================================================================================
//	LIBRARIES
//======================================================================================================================================================
#ifdef PROFILING
#include "RDTimer.h"
#endif

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <avilib.h>
#include <avimod.h>
#include <cuda.h>

#ifdef PROFILING
SimplePerfSerializer* serializeTime;
RDTimer* mytimer;
#endif

//======================================================================================================================================================
//	STRUCTURES, GLOBAL STRUCTURE VARIABLES
//======================================================================================================================================================

#include "define.c"

params_common_change common_change;
#ifdef USE_CONSTANT_BUFFER 
__constant__ params_common_change d_common_change;
#endif

params_common common;
#ifdef USE_CONSTANT_BUFFER 
__constant__ params_common d_common;
#endif

params_unique h_unique[ALL_POINTS];		// cannot determine size dynamically so choose more than usually needed

#ifdef USE_CONSTANT_BUFFER 
__constant__ params_unique d_unique[ALL_POINTS];
#else
params_unique* d_unique;
#endif

//======================================================================================================================================================
// KERNEL CODE
//======================================================================================================================================================

#include "kernel.cu"





//	WRITE DATA FUNCTION
//===============================================================================================================================================================================================================200

void write_data( int frameNo,
			int frames_processed,
			int endoPoints,
			int* input_a,
			int* input_b,
			int epiPoints,
			int* input_2a,
			int* input_2b){

	//================================================================================80
	//	VARIABLES
	//================================================================================80

	int i,j;
	char c;

	//================================================================================80
	//	WRITE VALUES TO THE FILE
	//================================================================================80
      fprintf(stdout, "Total AVI Frames: %d\n", frameNo);	
      fprintf(stdout, "Frames Processed: %d\n", frames_processed);	
      fprintf(stdout, "endoPoints: %d\n", endoPoints);
      fprintf(stdout, "epiPoints: %d", epiPoints);
	  for(j=0; j<frames_processed;j++)
	  {
	    fprintf(stdout, "\n---Frame %d---",j);
	    fprintf(stdout, "\n--endo--\n",j);
	    for(i=0; i<endoPoints; i++){
	      fprintf(stdout, "%d\t", input_a[j+i*frameNo]);
	    }
	    fprintf(stdout, "\n");
	    for(i=0; i<endoPoints; i++){
	      // if(input_b[j*size+i] > 2000) input_b[j*size+i]=0;
	      fprintf(stdout, "%d\t", input_b[j+i*frameNo]);
	    }
	    fprintf(stdout, "\n--epi--\n",j);
	    for(i=0; i<epiPoints; i++){
	      //if(input_2a[j*size_2+i] > 2000) input_2a[j*size_2+i]=0;
	      fprintf(stdout, "%d\t", input_2a[j+i*frameNo]);
	    }
	    fprintf(stdout, "\n");
	    for(i=0; i<epiPoints; i++){
	      //if(input_2b[j*size_2+i] > 2000) input_2b[j*size_2+i]=0;
	      fprintf(stdout, "%d\t", input_2b[j+i*frameNo]);
	    }
	  }
	// 	================================================================================80
	//		CLOSE FILE
	//	================================================================================80

	fflush(stdout);
	fclose(stdout);
}


//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
int main(int argc, char *argv []){

#ifdef PROFILING
	serializeTime = new SimplePerfSerializer( argv[0] );
	mytimer = new RDTimerGPU();
#endif

  fprintf(stderr, "WG size of kernel = %d \n", NUMBER_THREADS);
	//======================================================================================================================================================
	//	VARIABLES
	//======================================================================================================================================================

	// CUDA kernel execution parameters
	dim3 threads;
	dim3 blocks;

	// counter
	int i;
	int frames_processed;

	// frames
	char* video_file_name;
	avi_t* frames;
	fp* frame;

	//======================================================================================================================================================
	// 	FRAME
	//======================================================================================================================================================

	if(argc!=3){
		fprintf(stderr, "ERROR: usage: heartwall <inputfile> <num of frames>\n");
		exit(1);
	}
	
	// open movie file
 	video_file_name = argv[1];
	frames = (avi_t*)AVI_open_input_file(video_file_name, 1);														// added casting
	if (frames == NULL)  {
		   AVI_print_error((char *) "Error with AVI_open_input_file");
		   return -1;
	}

	// common
	common.no_frames = AVI_video_frames(frames);
	common.frame_rows = AVI_video_height(frames);
	common.frame_cols = AVI_video_width(frames);
	common.frame_elem = common.frame_rows * common.frame_cols;
	common.frame_mem = sizeof(fp) * common.frame_elem;

	// pointers
	hipMalloc((void **)&common_change.d_frame, common.frame_mem);

	//======================================================================================================================================================
	// 	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================
	
	frames_processed = atoi(argv[2]);
		if(frames_processed<0 || frames_processed>common.no_frames){
			fprintf(stdout, "ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n", frames_processed, common.no_frames);
			return 0;
	}
	

	//======================================================================================================================================================
	//	HARDCODED INPUTS FROM MATLAB
	//======================================================================================================================================================

	//====================================================================================================
	//	CONSTANTS
	//====================================================================================================

	common.sSize = 40;
	common.tSize = 25;
	common.maxMove = 10;
	common.alpha = 0.87;

	//====================================================================================================
	//	ENDO POINTS
	//====================================================================================================

	common.endoPoints = ENDO_POINTS;
	common.endo_mem = sizeof(int) * common.endoPoints;

	common.endoRow = (int *)malloc(common.endo_mem);
	common.endoRow[ 0] = 369;
	common.endoRow[ 1] = 400;
	common.endoRow[ 2] = 429;
	common.endoRow[ 3] = 452;
	common.endoRow[ 4] = 476;
	common.endoRow[ 5] = 486;
	common.endoRow[ 6] = 479;
	common.endoRow[ 7] = 458;
	common.endoRow[ 8] = 433;
	common.endoRow[ 9] = 404;
	common.endoRow[10] = 374;
	common.endoRow[11] = 346;
	common.endoRow[12] = 318;
	common.endoRow[13] = 294;
	common.endoRow[14] = 277;
	common.endoRow[15] = 269;
	common.endoRow[16] = 275;
	common.endoRow[17] = 287;
	common.endoRow[18] = 311;
	common.endoRow[19] = 339;
	hipMalloc((void **)&common.d_endoRow, common.endo_mem);
	hipMemcpy(common.d_endoRow, common.endoRow, common.endo_mem, hipMemcpyHostToDevice);

	common.endoCol = (int *)malloc(common.endo_mem);
	common.endoCol[ 0] = 408;
	common.endoCol[ 1] = 406;
	common.endoCol[ 2] = 397;
	common.endoCol[ 3] = 383;
	common.endoCol[ 4] = 354;
	common.endoCol[ 5] = 322;
	common.endoCol[ 6] = 294;
	common.endoCol[ 7] = 270;
	common.endoCol[ 8] = 250;
	common.endoCol[ 9] = 237;
	common.endoCol[10] = 235;
	common.endoCol[11] = 241;
	common.endoCol[12] = 254;
	common.endoCol[13] = 273;
	common.endoCol[14] = 300;
	common.endoCol[15] = 328;
	common.endoCol[16] = 356;
	common.endoCol[17] = 383;
	common.endoCol[18] = 401;
	common.endoCol[19] = 411;
	hipMalloc((void **)&common.d_endoCol, common.endo_mem);
	hipMemcpy(common.d_endoCol, common.endoCol, common.endo_mem, hipMemcpyHostToDevice);

	common.tEndoRowLoc = (int *)malloc(common.endo_mem * common.no_frames);
	hipMalloc((void **)&common.d_tEndoRowLoc, common.endo_mem * common.no_frames);

	common.tEndoColLoc = (int *)malloc(common.endo_mem * common.no_frames);
	hipMalloc((void **)&common.d_tEndoColLoc, common.endo_mem * common.no_frames);

	//====================================================================================================
	//	EPI POINTS
	//====================================================================================================

	common.epiPoints = EPI_POINTS;
	common.epi_mem = sizeof(int) * common.epiPoints;

	common.epiRow = (int *)malloc(common.epi_mem);
	common.epiRow[ 0] = 390;
	common.epiRow[ 1] = 419;
	common.epiRow[ 2] = 448;
	common.epiRow[ 3] = 474;
	common.epiRow[ 4] = 501;
	common.epiRow[ 5] = 519;
	common.epiRow[ 6] = 535;
	common.epiRow[ 7] = 542;
	common.epiRow[ 8] = 543;
	common.epiRow[ 9] = 538;
	common.epiRow[10] = 528;
	common.epiRow[11] = 511;
	common.epiRow[12] = 491;
	common.epiRow[13] = 466;
	common.epiRow[14] = 438;
	common.epiRow[15] = 406;
	common.epiRow[16] = 376;
	common.epiRow[17] = 347;
	common.epiRow[18] = 318;
	common.epiRow[19] = 291;
	common.epiRow[20] = 275;
	common.epiRow[21] = 259;
	common.epiRow[22] = 256;
	common.epiRow[23] = 252;
	common.epiRow[24] = 252;
	common.epiRow[25] = 257;
	common.epiRow[26] = 266;
	common.epiRow[27] = 283;
	common.epiRow[28] = 305;
	common.epiRow[29] = 331;
	common.epiRow[30] = 360;
	hipMalloc((void **)&common.d_epiRow, common.epi_mem);
	hipMemcpy(common.d_epiRow, common.epiRow, common.epi_mem, hipMemcpyHostToDevice);

	common.epiCol = (int *)malloc(common.epi_mem);
	common.epiCol[ 0] = 457;
	common.epiCol[ 1] = 454;
	common.epiCol[ 2] = 446;
	common.epiCol[ 3] = 431;
	common.epiCol[ 4] = 411;
	common.epiCol[ 5] = 388;
	common.epiCol[ 6] = 361;
	common.epiCol[ 7] = 331;
	common.epiCol[ 8] = 301;
	common.epiCol[ 9] = 273;
	common.epiCol[10] = 243;
	common.epiCol[11] = 218;
	common.epiCol[12] = 196;
	common.epiCol[13] = 178;
	common.epiCol[14] = 166;
	common.epiCol[15] = 157;
	common.epiCol[16] = 155;
	common.epiCol[17] = 165;
	common.epiCol[18] = 177;
	common.epiCol[19] = 197;
	common.epiCol[20] = 218;
	common.epiCol[21] = 248;
	common.epiCol[22] = 276;
	common.epiCol[23] = 304;
	common.epiCol[24] = 333;
	common.epiCol[25] = 361;
	common.epiCol[26] = 391;
	common.epiCol[27] = 415;
	common.epiCol[28] = 434;
	common.epiCol[29] = 448;
	common.epiCol[30] = 455;
	hipMalloc((void **)&common.d_epiCol, common.epi_mem);
	hipMemcpy(common.d_epiCol, common.epiCol, common.epi_mem, hipMemcpyHostToDevice);

	common.tEpiRowLoc = (int *)malloc(common.epi_mem * common.no_frames);
	hipMalloc((void **)&common.d_tEpiRowLoc, common.epi_mem * common.no_frames);

	common.tEpiColLoc = (int *)malloc(common.epi_mem * common.no_frames);
	hipMalloc((void **)&common.d_tEpiColLoc, common.epi_mem * common.no_frames);

	//====================================================================================================
	//	ALL POINTS
	//====================================================================================================

	common.allPoints = ALL_POINTS;

	//======================================================================================================================================================
	// 	TEMPLATE SIZES
	//======================================================================================================================================================

	// common
	common.in_rows = common.tSize + 1 + common.tSize;
	common.in_cols = common.in_rows;
	common.in_elem = common.in_rows * common.in_cols;
	common.in_mem = sizeof(fp) * common.in_elem;

	//======================================================================================================================================================
	// 	CREATE ARRAY OF TEMPLATES FOR ALL POINTS
	//======================================================================================================================================================

	// common
	hipMalloc((void **)&common.d_endoT, common.in_mem * common.endoPoints);
	hipMalloc((void **)&common.d_epiT, common.in_mem * common.epiPoints);

	//======================================================================================================================================================
	//	SPECIFIC TO ENDO OR EPI TO BE SET HERE
	//======================================================================================================================================================


	for(i=0; i<common.endoPoints; i++){
		h_unique[i].point_no = i;
		h_unique[i].d_Row = common.d_endoRow;
		h_unique[i].d_Col = common.d_endoCol;
		h_unique[i].d_tRowLoc = common.d_tEndoRowLoc;
		h_unique[i].d_tColLoc = common.d_tEndoColLoc;
		h_unique[i].d_T = common.d_endoT;
	}
	for(i=common.endoPoints; i<common.allPoints; i++){
		h_unique[i].point_no = i-common.endoPoints;
		h_unique[i].d_Row = common.d_epiRow;
		h_unique[i].d_Col = common.d_epiCol;
		h_unique[i].d_tRowLoc = common.d_tEpiRowLoc;
		h_unique[i].d_tColLoc = common.d_tEpiColLoc;
		h_unique[i].d_T = common.d_epiT;
	}

#ifndef USE_CONSTANT_BUFFER 
	hipMalloc((void**)&d_unique, ALL_POINTS*sizeof(params_unique));
#endif
	
	//======================================================================================================================================================
	// 	RIGHT TEMPLATE 	FROM 	TEMPLATE ARRAY
	//======================================================================================================================================================

	// pointers
	for(i=0; i<common.allPoints; i++){
		h_unique[i].in_pointer = h_unique[i].point_no * common.in_elem;
	}

	//======================================================================================================================================================
	// 	AREA AROUND POINT		FROM	FRAME
	//======================================================================================================================================================

	// common
	common.in2_rows = 2 * common.sSize + 1;
	common.in2_cols = 2 * common.sSize + 1;
	common.in2_elem = common.in2_rows * common.in2_cols;
	common.in2_mem = sizeof(float) * common.in2_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_in2, common.in2_mem);
	}

	//======================================================================================================================================================
	// 	CONVOLUTION
	//======================================================================================================================================================

	// common
	common.conv_rows = common.in_rows + common.in2_rows - 1;												// number of rows in I
	common.conv_cols = common.in_cols + common.in2_cols - 1;												// number of columns in I
	common.conv_elem = common.conv_rows * common.conv_cols;												// number of elements
	common.conv_mem = sizeof(float) * common.conv_elem;
	common.ioffset = 0;
	common.joffset = 0;

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_conv, common.conv_mem);
	}

	//======================================================================================================================================================
	// 	CUMULATIVE SUM
	//======================================================================================================================================================

	//====================================================================================================
	// 	PADDING OF ARRAY, VERTICAL CUMULATIVE SUM
	//====================================================================================================

	// common
	common.in2_pad_add_rows = common.in_rows;
	common.in2_pad_add_cols = common.in_cols;

	common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
	common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
	common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
	common.in2_pad_cumv_mem = sizeof(float) * common.in2_pad_cumv_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_in2_pad_cumv, common.in2_pad_cumv_mem);
	}

	//====================================================================================================
	// 	SELECTION
	//====================================================================================================

	// common
	common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;													// (1 to n+1)
	common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
	common.in2_pad_cumv_sel_collow = 1;
	common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
	common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
	common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
	common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
	common.in2_pad_cumv_sel_mem = sizeof(float) * common.in2_pad_cumv_sel_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_in2_pad_cumv_sel, common.in2_pad_cumv_sel_mem);
	}

	//====================================================================================================
	// 	SELECTION	2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
	//====================================================================================================

	// common
	common.in2_pad_cumv_sel2_rowlow = 1;
	common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
	common.in2_pad_cumv_sel2_collow = 1;
	common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
	common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
	common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
	common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
	common.in2_sub_cumh_mem = sizeof(float) * common.in2_sub_cumh_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_in2_sub_cumh, common.in2_sub_cumh_mem);
	}

	//====================================================================================================
	// 	SELECTION
	//====================================================================================================

	// common
	common.in2_sub_cumh_sel_rowlow = 1;
	common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
	common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
	common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
	common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
	common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
	common.in2_sub_cumh_sel_mem = sizeof(float) * common.in2_sub_cumh_sel_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_in2_sub_cumh_sel, common.in2_sub_cumh_sel_mem);
	}

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	// common
	common.in2_sub_cumh_sel2_rowlow = 1;
	common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel2_collow = 1;
	common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
	common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
	common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
	common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
	common.in2_sub2_mem = sizeof(float) * common.in2_sub2_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_in2_sub2, common.in2_sub2_mem);
	}

	//======================================================================================================================================================
	//	CUMULATIVE SUM 2
	//======================================================================================================================================================

	//====================================================================================================
	//	MULTIPLICATION
	//====================================================================================================

	// common
	common.in2_sqr_rows = common.in2_rows;
	common.in2_sqr_cols = common.in2_cols;
	common.in2_sqr_elem = common.in2_elem;
	common.in2_sqr_mem = common.in2_mem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_in2_sqr, common.in2_sqr_mem);
	}

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	// common
	common.in2_sqr_sub2_rows = common.in2_sub2_rows;
	common.in2_sqr_sub2_cols = common.in2_sub2_cols;
	common.in2_sqr_sub2_elem = common.in2_sub2_elem;
	common.in2_sqr_sub2_mem = common.in2_sub2_mem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_in2_sqr_sub2, common.in2_sqr_sub2_mem);
	}

	//======================================================================================================================================================
	//	FINAL
	//======================================================================================================================================================

	// common
	common.in_sqr_rows = common.in_rows;
	common.in_sqr_cols = common.in_cols;
	common.in_sqr_elem = common.in_elem;
	common.in_sqr_mem = common.in_mem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_in_sqr, common.in_sqr_mem);
	}

	//======================================================================================================================================================
	//	TEMPLATE MASK CREATE
	//======================================================================================================================================================

	// common
	common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
	common.tMask_cols = common.tMask_rows;
	common.tMask_elem = common.tMask_rows * common.tMask_cols;
	common.tMask_mem = sizeof(float) * common.tMask_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_tMask, common.tMask_mem);
	}

	//======================================================================================================================================================
	//	POINT MASK INITIALIZE
	//======================================================================================================================================================

	// common
	common.mask_rows = common.maxMove;
	common.mask_cols = common.mask_rows;
	common.mask_elem = common.mask_rows * common.mask_cols;
	common.mask_mem = sizeof(float) * common.mask_elem;

	//======================================================================================================================================================
	//	MASK CONVOLUTION
	//======================================================================================================================================================

	// common
	common.mask_conv_rows = common.tMask_rows;												// number of rows in I
	common.mask_conv_cols = common.tMask_cols;												// number of columns in I
	common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;												// number of elements
	common.mask_conv_mem = sizeof(float) * common.mask_conv_elem;
	common.mask_conv_ioffset = (common.mask_rows-1)/2;
	if((common.mask_rows-1) % 2 > 0.5){
		common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
	}
	common.mask_conv_joffset = (common.mask_cols-1)/2;
	if((common.mask_cols-1) % 2 > 0.5){
		common.mask_conv_joffset = common.mask_conv_joffset + 1;
	}

	// pointers
	for(i=0; i<common.allPoints; i++){
		hipMalloc((void **)&h_unique[i].d_mask_conv, common.mask_conv_mem);
	}

	//======================================================================================================================================================
	//	KERNEL
	//======================================================================================================================================================

	//====================================================================================================
	//	THREAD BLOCK
	//====================================================================================================

	// All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
	threads.x = NUMBER_THREADS;											// define the number of threads in the block
	threads.y = 1;
	blocks.x = common.allPoints;							// define the number of blocks in the grid
	blocks.y = 1;

	//====================================================================================================
	//	COPY ARGUMENTS
	//====================================================================================================
#ifdef USE_CONSTANT_BUFFER
	hipMemcpyToSymbol(d_common, &common, sizeof(params_common));
	hipMemcpyToSymbol(d_unique, &h_unique, sizeof(params_unique)*ALL_POINTS);
#else
	// pointers
//	for(i=0; i<common.allPoints; i++){
		hipMemcpy(d_unique, &h_unique, sizeof(params_unique)*ALL_POINTS, hipMemcpyHostToDevice);
//	}
#endif	
	
	//====================================================================================================
	//	PRINT FRAME PROGRESS START
	//====================================================================================================

	fprintf(stderr, "frame progress: ");
	fflush(NULL);

	//====================================================================================================
	//	LAUNCH
	//====================================================================================================

#ifdef PROFILING
	mytimer->Reset("Kernel");
#endif
	for(common_change.frame_no=0; common_change.frame_no<frames_processed; common_change.frame_no++){

		// Extract a cropped version of the first frame from the video file
		frame = get_frame(	frames,						// pointer to video file
										common_change.frame_no,				// number of frame that needs to be returned
										0,								// cropped?
										0,								// scaled?
										1);							// converted

		// copy frame to GPU memory
		hipMemcpy(common_change.d_frame, frame, common.frame_mem, hipMemcpyHostToDevice);
#ifdef PROFILING	
		mytimer->Start();
#endif
#ifdef USE_CONSTANT_BUFFER
		hipMemcpyToSymbol(d_common_change, &common_change, sizeof(params_common_change));

		// launch GPU kernel
		hipLaunchKernel(kernel, dim3(blocks), dim3(threads), 0, 0, );
#else

        hipLaunchKernel(kernel, dim3(blocks), dim3(threads), 0, 0, common, common_change, d_unique);
#endif
#ifdef PROFILING
		mytimer->Accumulate();
#endif
		// free frame after each loop iteration, since AVI library allocates memory for every frame fetched
		free(frame);

		// print frame progress
		fprintf( stderr, "%d ", common_change.frame_no);
		fflush(NULL);

	}

#ifdef PROFILING
	serializeTime->Serialize(mytimer);
#endif

	//====================================================================================================
	//	PRINT FRAME PROGRESS END
	//====================================================================================================

	fprintf(stderr, "\n");
	fflush(NULL);

	//====================================================================================================
	//	OUTPUT
	//====================================================================================================

	hipMemcpy(common.tEndoRowLoc, common.d_tEndoRowLoc, common.endo_mem * common.no_frames, hipMemcpyDeviceToHost);
	hipMemcpy(common.tEndoColLoc, common.d_tEndoColLoc, common.endo_mem * common.no_frames, hipMemcpyDeviceToHost);

	hipMemcpy(common.tEpiRowLoc, common.d_tEpiRowLoc, common.epi_mem * common.no_frames, hipMemcpyDeviceToHost);
	hipMemcpy(common.tEpiColLoc, common.d_tEpiColLoc, common.epi_mem * common.no_frames, hipMemcpyDeviceToHost);



#ifdef OUTPUT

	//==================================================50
	//	DUMP DATA TO FILE
	//==================================================50
	write_data(	common.no_frames,
				frames_processed,		
				common.endoPoints,
				common.tEndoRowLoc,
				common.tEndoColLoc,
				common.epiPoints,
				common.tEpiRowLoc,
				common.tEpiColLoc);

	//==================================================50
	//	End
	//==================================================50

#endif



	//======================================================================================================================================================
	//	DEALLOCATION
	//======================================================================================================================================================

	//====================================================================================================
	//	COMMON
	//====================================================================================================

	// frame
	hipFree(common_change.d_frame);

	// endo points
	free(common.endoRow);
	free(common.endoCol);
	free(common.tEndoRowLoc);
	free(common.tEndoColLoc);

	hipFree(common.d_endoRow);
	hipFree(common.d_endoCol);
	hipFree(common.d_tEndoRowLoc);
	hipFree(common.d_tEndoColLoc);

	hipFree(common.d_endoT);

	// epi points
	free(common.epiRow);
	free(common.epiCol);
	free(common.tEpiRowLoc);
	free(common.tEpiColLoc);

	hipFree(common.d_epiRow);
	hipFree(common.d_epiCol);
	hipFree(common.d_tEpiRowLoc);
	hipFree(common.d_tEpiColLoc);

	hipFree(common.d_epiT);

	//====================================================================================================
	//	POINTERS
	//====================================================================================================

	for(i=0; i<common.allPoints; i++){
		hipFree(h_unique[i].d_in2);

		hipFree(h_unique[i].d_conv);
		hipFree(h_unique[i].d_in2_pad_cumv);
		hipFree(h_unique[i].d_in2_pad_cumv_sel);
		hipFree(h_unique[i].d_in2_sub_cumh);
		hipFree(h_unique[i].d_in2_sub_cumh_sel);
		hipFree(h_unique[i].d_in2_sub2);
		hipFree(h_unique[i].d_in2_sqr);
		hipFree(h_unique[i].d_in2_sqr_sub2);
		hipFree(h_unique[i].d_in_sqr);

		hipFree(h_unique[i].d_tMask);
		hipFree(h_unique[i].d_mask_conv);
	}

#ifdef PROFILING
	delete serializeTime;
	delete mytimer;
#endif
}

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
