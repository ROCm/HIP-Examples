//====================================================================================================100
//	DEFINE / INCLUDE
//====================================================================================================100

#include "kernel_ecc.cu"
#include "kernel_cam.cu"
#include "kernel_fin.cu"
#include "kernel.cu"
#include "master.cu"
#include "embedded_fehlberg_7_8.cu"
#include "solver.cu"

//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100

int work(	int xmax,
				int workload){

	//================================================================================80
	//		VARIABLES
	//================================================================================80

	//============================================================60
	//		TIME
	//============================================================60

	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;

	time0 = get_time();

	//============================================================60
	//		COUNTERS
	//============================================================60

	long long memory;
	int i,j;
	int status;

	//============================================================60
	//		DATA
	//============================================================60

	fp*** y;
	fp** x;
	fp** params;
	fp* com;

	time1 = get_time();

	//================================================================================80
	// 	ALLOCATE MEMORY
	//================================================================================80

	//============================================================60
	//		MEMORY CHECK
	//============================================================60

	memory = workload*(xmax+1)*EQUATIONS*4;
	if(memory>1000000000){
		printf("ERROR: trying to allocate more than 1.0GB of memory, decrease workload and span parameters or change memory parameter\n");
		return 0;
	}

	//============================================================60
	// 	ALLOCATE ARRAYS
	//============================================================60

	y = (fp ***) malloc(workload* sizeof(fp **));
	for(i=0; i<workload; i++){
		y[i] = (fp**)malloc((1+xmax)*sizeof(fp*));
		for(j=0; j<(1+xmax); j++){
			y[i][j]= (fp *) malloc(EQUATIONS* sizeof(fp));
		}
	}

	x = (fp **) malloc(workload * sizeof(fp *));
	for (i= 0; i<workload; i++){
		x[i]= (fp *)malloc((1+xmax) *sizeof(fp));
	}

	params = (fp **) malloc(workload * sizeof(fp *));
	for (i= 0; i<workload; i++){
		params[i]= (fp *)malloc(PARAMETERS * sizeof(fp));
	}

	com = (fp*)malloc(3 * sizeof(fp));

	//============================================================60
	// 	ALLOCATE CUDA ARRAYS
	//============================================================60

	int d_initvalu_mem;
	d_initvalu_mem = EQUATIONS * sizeof(fp);
	fp* d_initvalu;
	cudaMalloc((void **)&d_initvalu, d_initvalu_mem);

	int d_finavalu_mem;
	d_finavalu_mem = EQUATIONS * sizeof(fp);
	fp* d_finavalu;
	cudaMalloc((void **)&d_finavalu, d_finavalu_mem);

	int d_params_mem;
	d_params_mem = PARAMETERS * sizeof(fp);
	fp* d_params;
	cudaMalloc((void **)&d_params, d_params_mem);

	int d_com_mem;
	d_com_mem = 3 * sizeof(fp);
	fp* d_com;
	cudaMalloc((void **)&d_com, d_com_mem);

	time2 = get_time();

	//================================================================================80
	// 	INITIAL VALUES
	//================================================================================80

	// y
	for(i=0; i<workload; i++){
		read(	"../../data/myocyte/y.txt",
					y[i][0],
					EQUATIONS,
					1,
					0);
	}

	// params
	for(i=0; i<workload; i++){
		read("../../data/myocyte/params.txt",
					params[i],
					PARAMETERS,
					1,
					0);
	}

	time3 = get_time();

	//================================================================================80
	//	EXECUTION
	//================================================================================80

	for(i=0; i<workload; i++){

		status = solver(	y[i],
									x[i],
									xmax,
									params[i],
									com,

									d_initvalu,
									d_finavalu,
									d_params,
									d_com);

		if(status !=0){
			printf("STATUS: %d\n", status);
		}

	}


	  FILE * pFile;
	  pFile = fopen ("output.txt","w");
	  if (pFile==NULL)
	    {
	  fputs ("fopen example",pFile);
	  return -1;
	}
	  // print results
	  int k;
	  for(i=0; i<workload; i++){
	  fprintf(pFile, "WORKLOAD %d:\n", i);
	  for(j=0; j<(xmax+1); j++){
	  fprintf(pFile, "\tTIME %d:\n", j);
	  for(k=0; k<EQUATIONS; k++){
	  fprintf(pFile, "\t\ty[%d][%d][%d]=%10.7e\n", i, j, k, y[i][j][k]);
	}
	}
	}

	  fclose (pFile);



	time4 = get_time();

	//================================================================================80
	//	DEALLOCATION
	//================================================================================80

	// y values
	for (i= 0; i< workload; i++){
		for (j= 0; j< (1+xmax); j++){
			free(y[i][j]);
		}
		free(y[i]);
	}
	free(y);

	// x values
	for (i= 0; i< workload; i++){
		free(x[i]);
	}
	free(x);

	// parameters
	for (i= 0; i< workload; i++){
		free(params[i]);
	}
	free(params);

	time5= get_time();

	// com
	free(com);

	// GPU memory
	cudaFree(d_initvalu);
	cudaFree(d_finavalu);
	cudaFree(d_params);
	cudaFree(d_com);

	//================================================================================80
	//		DISPLAY TIMING
	//================================================================================80

	printf("Time spent in different stages of the application:\n");
	printf("%.12f s, %.12f % : SETUP VARIABLES\n", 															(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time5-time0) * 100);
	printf("%.12f s, %.12f % : ALLOCATE CPU MEMORY AND GPU MEMORY\n", 				(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time5-time0) * 100);
	printf("%.12f s, %.12f % : READ DATA FROM FILES\n", 												(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time5-time0) * 100);
	printf("%.12f s, %.12f % : RUN COMPUTATION\n", 														(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time5-time0) * 100);
	printf("%.12f s, %.12f % : FREE MEMORY\n", 																(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time5-time0) * 100);
	printf("Total time:\n");
	printf("%.12f s\n", 																											(float) (time5-time0) / 1000000);

//====================================================================================================100
//	END OF FILE
//====================================================================================================100

	return 0;

}
