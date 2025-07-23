// XY model Metroporis sampler

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NX 32
#define NY 32

#define F(u, i, j) (u).f[(((u).ny) * ((i) - (u).iox) + ((j) - (u).ioy))]

typedef struct
{
  double *f;
  int nx, ny;
  int iox, ioy;
} field2d;

typedef struct
{
  int positive; // +1 vortices
  int negative; // -1 vortices
} vortex_count;

typedef struct
{
  double energy;
  double magnetization;
  double vortex_positive;
  double vortex_negative;
} sampling_stats;

// calculate the energy of the XY model (periodic boundary conditions)
double xyenergy(field2d u)
{
  double energy = 0.0;
  int i, j, ip, jp;

#pragma omp parallel for reduction(+ : energy) private(j, jp)
  for (i = 0; i < u.nx; i++)
  {
    ip = (i + 1) % u.nx;
    for (j = 0; j < u.ny; j++)
    {
      jp = (j + 1) % u.ny;
      energy += cos((F(u, i, j) - F(u, ip, j)) * 2.0 * M_PI); // Interaction with right neighbor
      energy += cos((F(u, i, j) - F(u, i, jp)) * 2.0 * M_PI); // Interaction with bottom neighbor
    }
  }

  return -energy / (u.nx * u.ny);
}

// calculate the magnetization of the XY model
double xymagnetization(field2d u)
{
  double magx = 0.0, magy = 0.0;
  int i, j;

#pragma omp parallel for reduction(+ : magx, magy) private(j)
  for (i = 0; i < u.nx; i++)
  {
    for (j = 0; j < u.ny; j++)
    {
      magx += cos(F(u, i, j) * 2.0 * M_PI); // X component
      magy += sin(F(u, i, j) * 2.0 * M_PI); // Y component
    }
  }

  return sqrt(magx * magx + magy * magy) / (u.nx * u.ny);
}

vortex_count xyvortex(field2d u)
{
  vortex_count result = {0, 0};
  int i, j;
  double theta[4], dtheta, total_phase;

  // Loop over all plaquettes (squares) in the lattice
#pragma omp parallel for private(j, theta, dtheta, total_phase) reduction(+ : result.positive, result.negative)
  for (i = 0; i < u.nx; i++)
  {
    for (j = 0; j < u.ny; j++)
    {
      // Get the four corner angles of the plaquette (in units of rad/2π)
      // Starting from bottom-left and going counter-clockwise
      theta[0] = F(u, i, j);                           // bottom-left
      theta[1] = F(u, (i + 1) % u.nx, j);              // bottom-right
      theta[2] = F(u, (i + 1) % u.nx, (j + 1) % u.ny); // top-right
      theta[3] = F(u, i, (j + 1) % u.ny);              // top-left

      // Calculate the total phase change around the plaquette
      total_phase = 0.0;
      for (int k = 0; k < 4; k++)
      {
        dtheta = theta[(k + 1) % 4] - theta[k];

        // Wrap the phase difference to [-0.5, 0.5] (equivalent to [-π, π])
        while (dtheta > 0.5)
          dtheta -= 1.0;
        while (dtheta < -0.5)
          dtheta += 1.0;

        total_phase += dtheta;
      }

      // Calculate the vorticity (winding number)
      int winding = (int)round(total_phase);

      // Count vortices by charge
      if (winding == 1)
      {
        result.positive++;
      }
      else if (winding == -1)
      {
        result.negative++;
      }
    }
  }

  return result;
}

// Initialize XY field with random values
void xy_initialize(field2d *u, int nx, int ny)
{
  int i, j;

  u->nx = nx;
  u->ny = ny;
  u->iox = 0;
  u->ioy = 0;
  u->f = (double *)malloc(u->nx * u->ny * sizeof(double));

  // Initialize the field with random values
#pragma omp parallel for private(j)
  for (i = 0; i < u->nx; i++)
  {
    for (j = 0; j < u->ny; j++)
    {
      F(*u, i, j) = (double)rand() / RAND_MAX; // Random initialization in [0, 1] range (0 to 2π rad)
    }
  }
}

// Perform XY model sampling with equilibration and statistics collection
sampling_stats xy_sampling(field2d *u, double beta, int nsamples, int verbose)
{
  int i, j, n, ip, jp, im, jm;
  double x, dx, x_new;
  double energy_old, energy_new, delta_energy;
  vortex_count vortices;
  sampling_stats stats = {0.0, 0.0, 0.0, 0.0};

  int equilibration_steps = 10000 * u->nx * u->ny; // 10000 MCS for equilibration
  int total_steps = equilibration_steps + nsamples * u->nx * u->ny;
  int sample_interval = u->nx * u->ny; // Sample every 1 MCS
  int samples_collected = 0;

  if (verbose)
  {
    printf("<metropolis_sampling_metadata\n");
    printf("equilibration_steps=%d\n", equilibration_steps);
    printf("nsamples=%d\n", nsamples);
    printf("metropolis_sampling_metadata>\n");

    printf("<metropolis_sampling\n");
  }
  // Metropolis sampling
  for (n = 0; n < total_steps; n++)
  {
    i = rand() % u->nx;
    j = rand() % u->ny;

    x = F(*u, i, j);
    dx = ((double)rand() / RAND_MAX - 0.5) * 0.1; // Small perturbation

    x_new = x + dx;

    // Keep angle in [0, 1] range (equivalent to [0, 2π))
    while (x_new >= 1.0)
      x_new -= 1.0;
    while (x_new < 0.0)
      x_new += 1.0;

    // Calculate energy difference (convert to radians for energy calculation)
    energy_old = 0.0;
    energy_new = 0.0;
    ip = (i + 1) % u->nx;
    im = (i - 1 + u->nx) % u->nx;
    jp = (j + 1) % u->ny;
    jm = (j - 1 + u->ny) % u->ny;

    // Old energy contribution
    energy_old += cos((x - F(*u, ip, j)) * 2.0 * M_PI);
    energy_old += cos((x - F(*u, im, j)) * 2.0 * M_PI);
    energy_old += cos((x - F(*u, i, jp)) * 2.0 * M_PI);
    energy_old += cos((x - F(*u, i, jm)) * 2.0 * M_PI);

    // New energy contribution
    energy_new += cos((x_new - F(*u, ip, j)) * 2.0 * M_PI);
    energy_new += cos((x_new - F(*u, im, j)) * 2.0 * M_PI);
    energy_new += cos((x_new - F(*u, i, jp)) * 2.0 * M_PI);
    energy_new += cos((x_new - F(*u, i, jm)) * 2.0 * M_PI);

    delta_energy = -(energy_new - energy_old);

    // Metropolis acceptance criterion
    if (delta_energy <= 0.0 || exp(-beta * delta_energy) > ((double)rand() / RAND_MAX))
    {
      F(*u, i, j) = x_new; // Accept the move
    }

    // Collect statistics after equilibration
    if (n >= equilibration_steps && (n - equilibration_steps) % sample_interval == 0 && samples_collected < nsamples)
    {
      vortices = xyvortex(*u);
      double energy = xyenergy(*u);
      double magnetization = xymagnetization(*u);

      stats.energy += energy;
      stats.magnetization += magnetization;
      stats.vortex_positive += vortices.positive;
      stats.vortex_negative += vortices.negative;

      samples_collected++;

      if (verbose)
        printf("%d %.3f %.3f %d %d\n", samples_collected, energy, magnetization, vortices.positive, vortices.negative);
    }
  }

  // Average the statistics
  if (samples_collected > 0)
  {
    stats.energy /= samples_collected;
    stats.magnetization /= samples_collected;
    stats.vortex_positive /= samples_collected;
    stats.vortex_negative /= samples_collected;
  }

  if (verbose)
    printf("metropolis_sampling>\n");

  return stats;
}
