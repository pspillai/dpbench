blackscholes_c_mpi_openmp:	blackscholes_c_mpi_openmp.c
	mpiicc -std=c99 -xCORE_AVX2 -fimf-precision"="high -D_VML_ACCURACY_HA_ -o blackscholes_c_mpi_openmp blackscholes_c_mpi_openmp.c -lm -qopenmp
