

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"

extern char *strcpy();
extern void exit();

int layer_size = 0;

void bpnn_save_dbg(net, filename)
BPNN *net;
char *filename;
{
  int n1, n2, n3, i, j;
  float **w;

  FILE *pFile;
  pFile = fopen( filename, "w+" );

  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  fprintf(pFile, "Saving %dx%dx%d network\n", n1, n2, n3);

  w = net->hidden_weights;
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
	  fprintf(pFile, "%d,%d,%f\n", i,j,w[i][j]);
    }
  }

  fclose(pFile);
  return;
}


backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err);
  bpnn_save_dbg(net, "out_hip.txt");
  bpnn_free(net);
  printf("Training done\n");
}

int setup(argc, argv)
int argc;
char *argv[];
{
	
  int seed;

  if (argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
