#include <stdio.h>
#include <omp.h>
#include "mpi.h"
#include <stdlib.h>
#include <math.h>
#include <mathimf.h>

static double * restrict price;
static double * restrict strike;
static double * restrict t;
static double * restrict put;
static double * restrict call;

double uniform(double a, double b, int *myseed) {
    double len = b - a;
    return ((double)rand_r(myseed) / RAND_MAX) * len + a;
}

int main(int argc, char **argv) {
  long    j;
  long    length,        /* vector length per rank                      */
          total_length=1000000000;  /* total vector length                         */
  double  local_nstream_time,/* timing parameters                       */
          nstream_time;
  int     Num_procs,     /* number of ranks                             */
          my_ID,         /* rank of calling rank                        */
          root=0;        /* ID of master rank                           */
  int     i, iters=1;    /* number of times to do calculations */

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&Num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_ID);

  if (argc < 2 || argc > 3){
    if (my_ID == root) 
      printf("Usage: %s <total length> <iterations>\n", *argv);
    exit(1);
  }
  total_length  = atol(*++argv);
  if (total_length < 1){
    if (my_ID == root) 
      printf("ERROR: total length must be > 0 : %d \n",total_length);
    exit(1);
  }
  if (argc>2){
    iters = atoi(*++argv);
    if (iters<1) {
      if (my_ID==root)
        printf("ERROR: iterations must be > 0: %d\n", iters);
      exit(1);
    }
  }

  if (my_ID == root) {
    length = total_length/Num_procs;
    printf("length: %d %d\n", length, Num_procs);
  }
  /* broadcast initialization data */
  MPI_Bcast(&length, 1, MPI_LONG, root, MPI_COMM_WORLD);

//  omp_set_num_threads(nthread_input);

  price = (double*)malloc(length * sizeof(double));
  strike = (double*)malloc(length * sizeof(double));
  t = (double*)malloc(length * sizeof(double));
  put = (double*)malloc(length * sizeof(double));
  call = (double*)malloc(length * sizeof(double));

  #pragma omp parallel 
  {
  unsigned int myseed = omp_get_thread_num();
  #pragma omp for
  for (j=0; j<length; j++) {
    price[j] = uniform(10.0, 50.0, &myseed);
    strike[j] = uniform(10.0, 50.0, &myseed);
    t[j] = uniform(1.0, 2.0, &myseed);
  }
  }

  if (my_ID == root) {
    printf("post initialization\n");
  }

  local_nstream_time = omp_get_wtime();
  double rate = 0.1;
  double vol = 0.2;
  double mr = -rate;
  double sig_sig_two = vol * vol * 2;

  for (i=0; i<iters; i++) {
    #pragma omp parallel for simd shared(price, strike, t, put, call)
    #pragma vector
    for (j=0; j<length; j++) {
      double P = price[j];
      double S = strike[j];
      double T = t[j];
      double a = log(P / S);
      double b = T * mr;
      double z = T * sig_sig_two;
      double c = 0.25 * z;
      double y = 1.0 / sqrt(z);
      double w1 = (a - b + c) * y;
      double w2 = (a - b - c) * y;
      //double d1 = 0.5 + 0.5 * erf(w1);
      //double d2 = 0.5 + 0.5 * erf(w2);
      double d1 = 0.5 + 0.5 * erff(w1);
      double d2 = 0.5 + 0.5 * erff(w2);
      double Se = exp(b) * S;
      call[j] = P * d1 - Se * d2;
      put[j] = call[j] - P + Se;
    }
  }

  local_nstream_time = omp_get_wtime() - local_nstream_time;
  MPI_Reduce(&local_nstream_time, &nstream_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

  if(my_ID == root) {
        printf("time: %f\n", nstream_time);
  }
  MPI_Finalize();
}
