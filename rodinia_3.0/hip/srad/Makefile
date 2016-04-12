include ../../common/make.config

$(TESTNAME): SRAD_V1 SRAD_V2

SRAD_V1:
	cd srad_v1; 	make

SRAD_V2:
	cd srad_v2; 	make

clean: SRAD_V1_clean SRAD_V2_clean
	rm -f image_out.pgm isa.txt *.out *.diff

SRAD_V1_clean:
	cd srad_v1; 	make clean
	
SRAD_V2_clean:
	cd srad_v2; 	make clean
