// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <omp.h>									// (in directory known to compiler)
#include <stdlib.h>									// (in directory known to compiler)

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"								// (in directory provided here)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/timer/timer.h"					// (in directory provided here)	needed by timer

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./kernel_cpu_2.h"							// (in directory provided here)

//========================================================================================================================================================================================================200
//	PLASMAKERNEL_GPU
//========================================================================================================================================================================================================200

void 
kernel_cpu_2(	int cores_arg,

				knode *knodes,
				long knodes_elem,

				int order,
				long maxheight,
				int count,

				long *currKnode,
				long *offset,
				long *lastKnode,
				long *offset_2,
				int *start,
				int *end,
				int *recstart,
				int *reclength)
{

	//======================================================================================================================================================150
	//	Variables
	//======================================================================================================================================================150

	// timer
	long long time0;
	long long time1;
	long long time2;

	// common variables
	int i;

	time0 = get_time();

	//======================================================================================================================================================150
	//	MCPU SETUP
	//======================================================================================================================================================150

	int max_nthreads;
	max_nthreads = omp_get_max_threads();
	// printf("max # of threads = %d\n", max_nthreads);
	omp_set_num_threads(cores_arg);
	// printf("set # of threads = %d\n", cores_arg);

	int threadsPerBlock;
	threadsPerBlock = order < 1024 ? order : 1024;

	time1 = get_time();

	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

	// private thread IDs
	int thid;
	int bid;

	// process number of querries
	#pragma omp parallel for private (i, thid)
	for(bid = 0; bid < count; bid++){

		// process levels of the tree
		for(i = 0; i < maxheight; i++){

			// process all leaves at each level
			for(thid = 0; thid < threadsPerBlock; thid++){

				if((knodes[currKnode[bid]].keys[thid] <= start[bid]) && (knodes[currKnode[bid]].keys[thid+1] > start[bid])){
					// this conditional statement is inserted to avoid crush due to but in original code
					// "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
					// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
					if(knodes[currKnode[bid]].indices[thid] < knodes_elem){
						offset[bid] = knodes[currKnode[bid]].indices[thid];
					}
				}
				if((knodes[lastKnode[bid]].keys[thid] <= end[bid]) && (knodes[lastKnode[bid]].keys[thid+1] > end[bid])){
					// this conditional statement is inserted to avoid crush due to but in original code
					// "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
					// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
					if(knodes[lastKnode[bid]].indices[thid] < knodes_elem){
						offset_2[bid] = knodes[lastKnode[bid]].indices[thid];
					}
				}

			}

			// set for next tree level
			currKnode[bid] = offset[bid];
			lastKnode[bid] = offset_2[bid];

		}

		// process leaves
		for(thid = 0; thid < threadsPerBlock; thid++){

			// Find the index of the starting record
			if(knodes[currKnode[bid]].keys[thid] == start[bid]){
				recstart[bid] = knodes[currKnode[bid]].indices[thid];
			}

		}

		// process leaves
		for(thid = 0; thid < threadsPerBlock; thid++){

			// Find the index of the ending record
			if(knodes[lastKnode[bid]].keys[thid] == end[bid]){
				reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid]+1;
			}

		}

	}

	time2 = get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

	printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",					(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time2-time0) * 100);
	printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time2-time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) (time2-time0) / 1000000);

} // main

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
