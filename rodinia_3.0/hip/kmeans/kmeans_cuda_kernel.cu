#include "hip/hip_runtime.h"
#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#include <stdio.h>
#include <cuda.h>

#include "kmeans.h"

// FIXME: Make this a runtime selectable variable!
#define ASSUMED_NR_CLUSTERS 32

#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

#ifdef USE_TEXTURES
// t_features has the layout dim0[points 0-m-1]dim1[ points 0-m-1]...
texture<float, 1, hipReadModeElementType> t_features;
// t_features_flipped has the layout point0[dim 0-n-1]point1[dim 0-n-1]
texture<float, 1, hipReadModeElementType> t_features_flipped;
texture<float, 1, hipReadModeElementType> t_clusters;
#endif


#ifdef USE_CONSTANT_BUFFER
#ifdef __KALMAR_CC__
// the initialization value 1e-8 is a workaround for a bug in HLC
// strangely speaking, the issue would only happen on Fiji, NOT on Kaveri
__attribute__((address_space(2))) float c_clusters[ASSUMED_NR_CLUSTERS*34] = { 1e-8 };		/* constant memory for cluster centers */
#else
__constant__ float c_clusters[ASSUMED_NR_CLUSTERS*34];		/* constant memory for cluster centers */
#endif
#endif

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
__global__ void 
invert_mapping(hipLaunchParm lp,
               float *input,			/* original */
               float *output,			/* inverted */
               int npoints,				/* npoints */
               int nfeatures)			/* nfeatures */
{
	int point_id = hipThreadIdx_x + hipBlockDim_x*hipBlockIdx_x;	/* id of thread */
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
 kmeansPoint(hipLaunchParm lp, 
            float  *features,			/* in: [npoints*nfeatures] */
            float  *features_flipped,
            int     nfeatures,
            int     npoints,
            int     nclusters,
            int    *membership,
			float  *clusters,
			float  *block_clusters,
			int    *block_deltas) 
{

	// block ID
	const unsigned int block_id = hipGridDim_x*hipBlockIdx_y+hipBlockIdx_x;
	// point/thread ID  
	const unsigned int point_id = block_id*hipBlockDim_x*hipBlockDim_y + hipThreadIdx_x;
  
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
#ifdef USE_TEXTURES
				float diff = (tex1Dfetch(t_features,addr) -
#else
				float diff = (features[addr] -
#endif
#ifdef USE_CONSTANT_BUFFER
							  c_clusters[cluster_base_index + j]);	/* distance between a data point to cluster centers */
#else
							  clusters[cluster_base_index + j]);	/* distance between a data point to cluster centers */
#endif
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
	if(hipThreadIdx_x < THREADS_PER_BLOCK) {
		deltas[hipThreadIdx_x] = 0;
	}
#endif
	if (point_id < npoints)
	{
#ifdef GPU_DELTA_REDUCTION
		/* if membership changes, increase delta by 1 */
		if (membership[point_id] != index) {
			deltas[hipThreadIdx_x] = 1;
		}
#endif
		/* assign the membership to object point_id */
		membership[point_id] = index;
	}

#ifdef GPU_DELTA_REDUCTION
	// make sure all the deltas have finished writing to shared memory
	idx.barrier.wait();

	// now let's count them
	// primitve reduction follows
	unsigned int threadids_participating = THREADS_PER_BLOCK / 2;
	for(;threadids_participating > 1; threadids_participating /= 2) {
   		if(hipThreadIdx_x < threadids_participating) {
			deltas[hipThreadIdx_x] += deltas[hipThreadIdx_x + threadids_participating];
		}
   		idx.barrier.wait();
	}
	if(hipThreadIdx_x < 1)	{deltas[hipThreadIdx_x] += deltas[hipThreadIdx_x + 1];}
	idx.barrier.wait();
		// propagate number of changes to global counter
	if(hipThreadIdx_x == 0) {
		block_deltas[hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x] = deltas[0];
		//printf("original id: %d, modified: %d\n", hipBlockIdx_y*hipGridDim_x+hipBlockIdx_x, hipBlockIdx_x);
		
	}

#endif


#ifdef GPU_NEW_CENTER_REDUCTION
	int center_id = hipThreadIdx_x / nfeatures;    
	int dim_id = hipThreadIdx_x - nfeatures*center_id;

	__shared__ int new_center_ids[THREADS_PER_BLOCK];

	new_center_ids[hipThreadIdx_x] = index;
	idx.barrier.wait();

	/***
	determine which dimension calculte the sum for
	mapping of threads is
	center0[dim0,dim1,dim2,...]center1[dim0,dim1,dim2,...]...
	***/ 	

	int new_base_index = (point_id - hipThreadIdx_x)*nfeatures + dim_id;
	float accumulator = 0.f;

	if(hipThreadIdx_x < nfeatures * nclusters) {
		// accumulate over all the elements of this threadblock 
		for(int i = 0; i< (THREADS_PER_BLOCK); i++) {
#ifdef USE_TEXTURES
			float val = tex1Dfetch(t_features_flipped,new_base_index+i*nfeatures);
#else
			float val = features_flipped[new_base_index+i*nfeatures];
#endif
			if(new_center_ids[i] == center_id) 
				accumulator += val;
		}
	
		// now store the sum for this threadblock
		/***
		mapping to global array is
		block0[center0[dim0,dim1,dim2,...]center1[dim0,dim1,dim2,...]...]block1[...]...
		***/
		block_clusters[(hipBlockIdx_y*hipGridDim_x + hipBlockIdx_x) * nclusters * nfeatures + hipThreadIdx_x] = accumulator;
	}
#endif

}
#endif // #ifndef _KMEANS_CUDA_KERNEL_H_
