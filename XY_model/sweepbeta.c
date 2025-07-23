// XY model beta sweep program

#include "xymetoropolis.h"

int main(int argc, char **argv)
{
  field2d u;
  int nsamples;
  double beta_min = 0.1;
  double beta_max = 2.0;
  double beta_step = 0.05;
  sampling_stats stats;

  srand(42); // Fixed seed for reproducibility

  if (argc != 2)
  {
    fprintf(stderr, "Usage: %s <nsamples>\n", argv[0]);
    return 1;
  }

  nsamples = atoi(argv[1]);

  printf("beta_min=%.1f, beta_max=%.1f, beta_step=%.1f, nsamples=%d, nx=%d, ny=%d\n",
         beta_min, beta_max, beta_step, nsamples, NX, NY);

  printf("\n<beta_sweep_results\n");

  // Loop over beta values
  for (double beta = beta_min; beta <= beta_max + 1e-10; beta += beta_step)
  {
    // Initialize the field for each beta value
    xy_initialize(&u, NX, NY);

    // Perform sampling (verbose=0 to suppress detailed output)
    stats = xy_sampling(&u, beta, nsamples, 0);

    // Output in the requested format
    printf("%.2f %.6f %.6f %.3f %.3f\n",
           beta, stats.energy, stats.magnetization,
           stats.vortex_positive, stats.vortex_negative);

    // Free memory for this beta value
    free(u.f);
  }

  printf("beta_sweep_results>\n");

  return 0;
}
