#ifndef XYMETROPOLIS_H
#define XYMETROPOLIS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NX 32
#define NY 32

#define F(u, i, j) (u).f[(((u).ny) * ((i) - (u).iox) + ((j) - (u).ioy))]

// Structure to represent a 2D field
typedef struct
{
  double *f;
  int nx, ny;
  int iox, ioy;
} field2d;

// Structure to count vortices by charge
typedef struct
{
  int positive; // +1 vortices
  int negative; // -1 vortices
} vortex_count;

// Structure to store sampling statistics
typedef struct
{
  double energy;
  double magnetization;
  double vortex_positive;
  double vortex_negative;
} sampling_stats;

// Function declarations
double xyenergy(field2d u);
double xymagnetization(field2d u);
vortex_count xyvortex(field2d u);
void xy_initialize(field2d *u, int nx, int ny);
sampling_stats xy_sampling(field2d *u, double beta, int nsamples, int verbose);

#endif // XYMETROPOLIS_H
