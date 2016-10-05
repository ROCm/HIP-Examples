#include "hip/hip_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

//#include <omp.h>

#include <cuda.h>

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM

#include "kmeans_cuda_kernel.cu"


//#define BLOCK_DELTA_REDUCE
//#define BLOCK_CENTER_REDUCE

#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE

extern "C"
int setup(int argc, char** argv);									/* function prototype */

// GLOBAL!!!!!
unsigned int num_threads_perdim = THREADS_PER_DIM;					/* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;					/* temporary */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;	/* number of threads */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;		/* number of blocks */

/* _d denotes it resides on the device */
int    *membership_new;												/* newly assignment membership */
float  *feature_d;													/* inverted data array */
float  *feature_flipped_d;											/* original (not inverted) data array */
int    *membership_d;												/* membership on the device */
float  *block_new_centers;											/* sum of points in a cluster (per block) */
float  *clusters_d;													/* cluster centers on the device */
float  *block_clusters_d;											/* per block calculation of cluster centers */
int    *block_deltas_d;												/* per block calculation of deltas */


/* -------------- allocateMemory() ------------------- */
/* allocate device memory, calculate number of blocks and threads, and invert the data array */
extern "C"
void allocateMemory(int npoints, int nfeatures, int nclusters, float **features)
{	
	num_blocks = npoints / num_threads;
	if (npoints % num_threads > 0)		/* defeat truncation */
		num_blocks++;

	num_blocks_perdim = sqrt((double) num_blocks);
	while (num_blocks_perdim * num_blocks_perdim < num_blocks)	// defeat truncation (should run once)
		num_blocks_perdim++;

	num_blocks = num_blocks_perdim*num_blocks_perdim;

	/* allocate memory for memory_new[] and initialize to -1 (host) */
	membership_new = (int*) malloc(npoints * sizeof(int));
	for(int i=0;i<npoints;i++) {
		membership_new[i] = -1;
	}

	/* allocate memory for block_new_centers[] (host) */
	block_new_centers = (float *) malloc(nclusters*nfeatures*sizeof(float));
	
	/* allocate memory for feature_flipped_d[][], feature_d[][] (device) */
	hipMalloc((void**) &feature_flipped_d, npoints*nfeatures*sizeof(float));
	hipMemcpy(feature_flipped_d, features[0], npoints*nfeatures*sizeof(float), hipMemcpyHostToDevice);
	hipMalloc((void**) &feature_d, npoints*nfeatures*sizeof(float));
		
	/* invert the data array (kernel execution) */	
	hipLaunchKernel(invert_mapping, dim3(num_blocks), dim3(num_threads), 0, 0, feature_flipped_d,feature_d,npoints,nfeatures);
		
	/* allocate memory for membership_d[] and clusters_d[][] (device) */
	hipMalloc((void**) &membership_d, npoints*sizeof(int));
	hipMalloc((void**) &clusters_d, nclusters*nfeatures*sizeof(float));

	
#ifdef BLOCK_DELTA_REDUCE
	// allocate array to hold the per block deltas on the gpu side
	
	hipMalloc((void**) &block_deltas_d, num_blocks_perdim * num_blocks_perdim * sizeof(int));
	//hipMemcpy(block_delta_d, &delta_h, sizeof(int), hipMemcpyHostToDevice);
#endif

#ifdef BLOCK_CENTER_REDUCE
	// allocate memory and copy to card cluster  array in which to accumulate center points for the next iteration
	hipMalloc((void**) &block_clusters_d, 
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float));
	//hipMemcpy(new_clusters_d, new_centers[0], nclusters*nfeatures*sizeof(float), hipMemcpyHostToDevice);
#endif

}
/* -------------- allocateMemory() end ------------------- */

/* -------------- deallocateMemory() ------------------- */
/* free host and device memory */
extern "C"
void deallocateMemory()
{
	free(membership_new);
	free(block_new_centers);
	hipFree(feature_d);
	hipFree(feature_flipped_d);
	hipFree(membership_d);

	hipFree(clusters_d);
#ifdef BLOCK_CENTER_REDUCE
    hipFree(block_clusters_d);
#endif
#ifdef BLOCK_DELTA_REDUCE
    hipFree(block_deltas_d);
#endif
}
/* -------------- deallocateMemory() end ------------------- */



////////////////////////////////////////////////////////////////////////////////
// Program main																  //

int
main( int argc, char** argv) 
{
    int device_cnt;
    hipGetDeviceCount(&device_cnt);
    if (device_cnt > 1) {
        // make sure we're running on the big card, but only if it exists...
        hipError_t err = hipSetDevice(0);
        //printf ("setDevice: %d\n", err);
    } 
	// as done in the CUDA start/help document provided
	setup(argc, argv);    
}

//																			  //
////////////////////////////////////////////////////////////////////////////////


/* ------------------- kmeansCuda() ------------------------ */    
extern "C"
int	// delta -- had problems when return value was of float type
kmeansCuda(float  **feature,				/* in: [npoints][nfeatures] */
           int      nfeatures,				/* number of attributes for each point */
           int      npoints,				/* number of data points */
           int      nclusters,				/* number of clusters */
           int     *membership,				/* which cluster the point belongs to */
		   float  **clusters,				/* coordinates of cluster centers */
		   int     *new_centers_len,		/* number of elements in each cluster */
           float  **new_centers				/* sum of elements in each cluster */
		   )
{
	int delta = 0;			/* if point has moved */
	int i,j;				/* counters */


	//hipSetDevice(1);

	/* copy membership (host to device) */
	hipMemcpy(membership_d, membership_new, npoints*sizeof(int), hipMemcpyHostToDevice);

	/* copy clusters (host to device) */
	hipMemcpy(clusters_d, clusters[0], nclusters*nfeatures*sizeof(float), hipMemcpyHostToDevice);

#ifdef USE_TEXTURES
	/* set up texture */
    hipChannelFormatDesc chDesc0 = hipCreateChannelDesc<float>();
    t_features.filterMode = hipFilterModePoint;   
    t_features.normalized = false;
    t_features.channelDesc = chDesc0;

	if(hipBindTexture(NULL, t_features, feature_d, npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
        printf("Couldn't bind features array to texture!\n");

	hipChannelFormatDesc chDesc1 = hipCreateChannelDesc<float>();
    t_features_flipped.filterMode = hipFilterModePoint;   
    t_features_flipped.normalized = false;
    t_features_flipped.channelDesc = chDesc1;

	if(hipBindTexture(NULL, t_features_flipped, feature_flipped_d,  npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
        printf("Couldn't bind features_flipped array to texture!\n");

	hipChannelFormatDesc chDesc2 = hipCreateChannelDesc<float>();
    t_clusters.filterMode = hipFilterModePoint;   
    t_clusters.normalized = false;
    t_clusters.channelDesc = chDesc2;

	if(hipBindTexture(NULL, t_clusters, clusters_d,  nclusters*nfeatures*sizeof(float)) != CUDA_SUCCESS)
        printf("Couldn't bind clusters array to texture!\n");
#endif

#ifdef USE_CONSTANT_BUFFER
	/* copy clusters to constant memory */
	hipError_t err = hipMemcpyToSymbol("c_clusters",clusters[0],nclusters*nfeatures*sizeof(float),0,hipMemcpyHostToDevice);
    //printf ("cudaMemcpyToSymbol: %d (%s)\n", err, hipGetErrorString(err));
#endif


    /* setup execution parameters.
	   changed to 2d (source code on NVIDIA CUDA Programming Guide) */
    dim3  grid( num_blocks_perdim, num_blocks_perdim );
    dim3  threads( num_threads_perdim*num_threads_perdim );
    
	/* execute the kernel */
    hipLaunchKernel(kmeansPoint, dim3(grid), dim3(threads ), 0, 0,  feature_d,
                                      feature_flipped_d,
                                      nfeatures,
                                      npoints,
                                      nclusters,
                                      membership_d,
                                      clusters_d,
									  block_clusters_d,
									  block_deltas_d);

	hipDeviceSynchronize();

	/* copy back membership (device to host) */
	hipMemcpy(membership_new, membership_d, npoints*sizeof(int), hipMemcpyDeviceToHost);	

#ifdef BLOCK_CENTER_REDUCE
    /*** Copy back arrays of per block sums ***/
    float * block_clusters_h = (float *) malloc(
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float));
        
	hipMemcpy(block_clusters_h, block_clusters_d, 
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float), 
        hipMemcpyDeviceToHost);
#endif
#ifdef BLOCK_DELTA_REDUCE
    int * block_deltas_h = (int *) malloc(
        num_blocks_perdim * num_blocks_perdim * sizeof(int));
        
	hipMemcpy(block_deltas_h, block_deltas_d, 
        num_blocks_perdim * num_blocks_perdim * sizeof(int), 
        hipMemcpyDeviceToHost);
#endif
    
	/* for each point, sum data points in each cluster
	   and see if membership has changed:
	     if so, increase delta and change old membership, and update new_centers;
	     otherwise, update new_centers */
	delta = 0;
	for (i = 0; i < npoints; i++)
	{		
		int cluster_id = membership_new[i];
		new_centers_len[cluster_id]++;
		if (membership_new[i] != membership[i])
		{
#ifdef CPU_DELTA_REDUCE
			delta++;
#endif
			membership[i] = membership_new[i];
		}
#ifdef CPU_CENTER_REDUCE
		for (j = 0; j < nfeatures; j++)
		{			
			new_centers[cluster_id][j] += feature[i][j];
		}
#endif
	}
	

#ifdef BLOCK_DELTA_REDUCE	
    /*** calculate global sums from per block sums for delta and the new centers ***/    
	
	//debug
	//printf("\t \t reducing %d block sums to global sum \n",num_blocks_perdim * num_blocks_perdim);
    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		//printf("block %d delta is %d \n",i,block_deltas_h[i]);
        delta += block_deltas_h[i];
    }
        
#endif
#ifdef BLOCK_CENTER_REDUCE	
	
	for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++) {
			block_new_centers[j*nfeatures + k] = 0.f;
		}
	}

    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		for(int j = 0; j < nclusters;j++) {
			for(int k = 0; k < nfeatures;k++) {
				block_new_centers[j*nfeatures + k] += block_clusters_h[i * nclusters*nfeatures + j * nfeatures + k];
			}
		}
    }
	

#ifdef CPU_CENTER_REDUCE
	//debug
	/*for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++) {
			if(new_centers[j][k] >	1.001 * block_new_centers[j*nfeatures + k] || new_centers[j][k] <	0.999 * block_new_centers[j*nfeatures + k]) {
				printf("\t \t for %d:%d, normal value is %e and gpu reduced value id %e \n",j,k,new_centers[j][k],block_new_centers[j*nfeatures + k]);
			}
		}
	}*/
#endif

#ifdef BLOCK_CENTER_REDUCE
	for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++)
			new_centers[j][k]= block_new_centers[j*nfeatures + k];		
	}
#endif

#endif

	return delta;
	
}
/* ------------------- kmeansCuda() end ------------------------ */    

