all: OPENMP/kmeans SERIAL/kmeans

OPENMP/kmeans:
	cd kmeans_openmp; make
	
SERIAL/kmeans:
	cd kmeans_serial; make

clean:
	cd kmeans_serial; make clean
	cd kmeans_openmp; make clean
	
