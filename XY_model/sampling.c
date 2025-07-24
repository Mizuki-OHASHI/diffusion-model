#include "xymetoropolis.h"

int main(int argc, char **argv)
{
  field2d u;
  int i, j, nsamples, size;
  double beta;
  sampling_stats stats;

  srand(42);

  if (argc != 4)
  {
    fprintf(stderr, "Usage: %s <nsamples> <size> <beta>\n", argv[0]);
    return 1;
  }

  nsamples = atoi(argv[1]);
  size = atoi(argv[2]);
  beta = atof(argv[3]);

  printf("nsamples=%d, beta=%.2f, nx=%d, ny=%d\n", nsamples, beta, size, size);

  // Initialize the field
  xy_initialize(&u, size, size);

  // Perform sampling with equilibration
  stats = xy_sampling(&u, beta, nsamples, 1);

  // Output final statistics
  printf("\n<final_statistics_metadata\n");
  printf("energy=%.6f\n", stats.energy);
  printf("magnetization=%.6f\n", stats.magnetization);
  printf("vortex_positive=%.3f\n", stats.vortex_positive);
  printf("vortex_negative=%.3f\n", stats.vortex_negative);
  printf("final_statistics_metadata>\n");

  // Output the final field values (last sample)
  printf("\n<final_field_values\n");
  for (i = 0; i < u.nx; i++)
  {
    for (j = 0; j < u.ny; j++)
    {
      printf("%.3f ", F(u, i, j));
    }
    printf("\n");
  }
  printf("final_field_values>\n");

  free(u.f);
  return 0;
}
