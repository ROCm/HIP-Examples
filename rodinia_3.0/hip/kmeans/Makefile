include ../../common/make.config

# C compiler
CC = gcc
CFLAGS = -g -O2

#Use textures for features:
#CFLAGS_EX += -DUSE_TEXTURES
#Use constant buffer for c_clusters var:
#CFLAGS_EX += -DUSE_CONSTANT_BUFFER

CFLAGS    += $(CFLAGS_EX)
HIPCC_FLAGS += $(CFLAGS_EX)

#Use textures for features:
#HIPCC += -DUSE_TEXTURES
#Use constant buffer for c_clusters var:
#HIPCC += -DUSE_CONSTANT_BUFFER

ifeq ($(HIP_PLATFORM), hcc)
        #Use constant buffer for c_clusters var:
        #HIPCC += -DUSE_CONSTANT_BUFFER
endif

$(TESTNAME):cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o
	$(HIPLD) $(CFLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o -o kmeans -lm

kmeans_cuda.o: kmeans_cuda.cu  kmeans_cuda_kernel.cu $(HIP_DEPS) 
	$(HIPCC) $(HIPCC_FLAGS) -c kmeans_cuda.cu

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo
