#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#include <stdio.h>
#include <cuda.h>

#include "kmeans.h"

// FIXME: Make this a runtime selectable variable!
#define ASSUMED_NR_CLUSTERS 32

#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

// t_features has the layout dim0[points 0-m-1]dim1[ points 0-m-1]...
texture<float, 1, cudaReadModeElementType> t_features;
// t_features_flipped has the layout point0[dim 0-n-1]point1[dim 0-n-1]
texture<float, 1, cudaReadModeElementType> t_features_flipped;
texture<float, 1, cudaReadModeElementType> t_clusters;


__constant__ float c_clusters[ASSUMED_NR_CLUSTERS*34];		/* constant memory for cluster centers */

/* ----------------- invert_mapping() --------------------- */
/* inverts data array from row-major to column-major.

   [p0,dim0][p0,dim1][p0,dim2] ... 
   [p1,dim0][p1,dim1][p1,dim2] ... 
   [p2,dim0][p2,dim1][p2,dim2] ... 
										to
   [dim0,p0][dim0,p1][dim0,p2] ...
   [dim1,p0][dim1,p1][dim1,p2] ...
   [dim2,p0][dim2,p1][dim2,p2] ...
*/
__global__ void invert_mapping(float *input,			/* original */
							   float *output,			/* inverted */
							   int npoints,				/* npoints */
							   int nfeatures)			/* nfeatures */
{
	int point_id = threadIdx.x + blockDim.x*blockIdx.x;	/* id of thread */
	int i;

	if(point_id < npoints){
		for(i=0;i<nfeatures;i++)
			output[point_id + npoints*i] = input[point_id*nfeatures + i];
	}
	return;
}
/* ----------------- invert_mapping() end --------------------- */

/* to turn on the GPU delta and center reduction */
//#define GPU_DELTA_REDUCTION
//#define GPU_NEW_CENTER_REDUCTION


/* ----------------- kmeansPoint() --------------------- */
/* find the index of nearest cluster centers and change membership*/
__global__ void
kmeansPoint(float  *features,			/* in: [npoints*nfeatures] */
            int     nfeatures,
            int     npoints,
            int     nclusters,
            int    *membership,
			float  *clusters,
			float  *block_clusters,
			int    *block_deltas) 
{

	// block ID
	const unsigned int block_id = gridDim.x*blockIdx.y+blockIdx.x;
	// point/thread ID  
	const unsigned int point_id = block_id*blockDim.x*blockDim.y + threadIdx.x;
  
	int  index = -1;

	if (point_id < npoints)
	{
		int i, j;
		float min_dist = FLT_MAX;
		float dist;													/* distance square between a point to cluster center */
		
		/* find the cluster center id with min distance to pt */
		for (i=0; i<nclusters; i++) {
			int cluster_base_index = i*nfeatures;					/* base index of cluster centers for inverted array */			
			float ans=0.0;												/* Euclidean distance sqaure */

			for (j=0; j < nfeatures; j++)
			{					
				int addr = point_id + j*npoints;					/* appropriate index of data point */
				float diff = (tex1Dfetch(t_features,addr) -
							  c_clusters[cluster_base_index + j]);	/* distance between a data point to cluster centers */
				ans += diff*diff;									/* sum of squares */
			}
			dist = ans;		

			/* see if distance is smaller than previous ones:
			if so, change minimum distance and save index of cluster center */
			if (dist < min_dist) {
				min_dist = dist;
				index    = i;
			}
		}
	}
	

#ifdef GPU_DELTA_REDUCTION
    // count how many points are now closer to a different cluster center	
	__shared__ int deltas[THREADS_PER_BLOCK];
	if(threadIdx.x < THREADS_PER_BLOCK) {
		deltas[threadIdx.x] = 0;
	}
#endif
	if (point_id < npoints)
	{
#ifdef GPU_DELTA_REDUCTION
		/* if membership changes, increase delta by 1 */
		if (membership[point_id] != index) {
			deltas[threadIdx.x] = 1;
		}
#endif
		/* assign the membership to object point_id */
		membership[point_id] = index;
	}

#ifdef GPU_DELTA_REDUCTION
	// make sure all the deltas have finished writing to shared memory
	__syncthreads();

	// now let's count them
	// primitve reduction follows
	unsigned int threadids_participating = THREADS_PER_BLOCK / 2;
	for(;threadids_participating > 1; threadids_participating /= 2) {
   		if(threadIdx.x < threadids_participating) {
			deltas[threadIdx.x] += deltas[threadIdx.x + threadids_participating];
		}
   		__syncthreads();
	}
	if(threadIdx.x < 1)	{deltas[threadIdx.x] += deltas[threadIdx.x + 1];}
	__syncthreads();
		// propagate number of changes to global counter
	if(threadIdx.x == 0) {
		block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = deltas[0];
		//printf("original id: %d, modified: %d\n", blockIdx.y*gridDim.x+blockIdx.x, blockIdx.x);
		
	}

#endif


#ifdef GPU_NEW_CENTER_REDUCTION
	int center_id = threadIdx.x / nfeatures;    
	int dim_id = threadIdx.x - nfeatures*center_id;

	__shared__ int new_center_ids[THREADS_PER_BLOCK];

	new_center_ids[threadIdx.x] = index;
	__syncthreads();

	/***
	determine which dimension calculte the sum for
	mapping of threads is
	center0[dim0,dim1,dim2,...]center1[dim0,dim1,dim2,...]...
	***/ 	

	int new_base_index = (point_id - threadIdx.x)*nfeatures + dim_id;
	float accumulator = 0.f;

	if(threadIdx.x < nfeatures * nclusters) {
		// accumulate over all the elements of this threadblock 
		for(int i = 0; i< (THREADS_PER_BLOCK); i++) {
			float val = tex1Dfetch(t_features_flipped,new_base_index+i*nfeatures);
			if(new_center_ids[i] == center_id) 
				accumulator += val;
		}
	
		// now store the sum for this threadblock
		/***
		mapping to global array is
		block0[center0[dim0,dim1,dim2,...]center1[dim0,dim1,dim2,...]...]block1[...]...
		***/
		block_clusters[(blockIdx.y*gridDim.x + blockIdx.x) * nclusters * nfeatures + threadIdx.x] = accumulator;
	}
#endif

}
#endif // #ifndef _KMEANS_CUDA_KERNEL_H_
