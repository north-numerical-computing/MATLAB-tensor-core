#include "../include/gemm.h"
#include <stdint.h>
#include <math.h>
#include <string.h>

/* Conversion functions between uint64 and double */
static inline uint64_t d2u(double x) {
  uint64_t u;
  memcpy(&u, &x, 8);
  return u;
}

static inline double u2d(uint64_t u) {
  double x;
  memcpy(&x, &u, 8);
  return x;
}

/* Perform inner product a*b+c using a tensor core model configured
   by def_params.

   Inputs vectors of size def_params->fma are rounded to informat
   using cpfloat.

   Elementwise product a*b is assumed to be kept exactly.

   Output c is rounded to outformat using cpfloat.

   The computation is performed in the fixed-point arithmetic
   defined by various settings in def_params, rounded to outformat
   using cpfloat, and written to c.
 */
void block_FMA_nv(double *a, double *b, double *c, tc_params *def_params) {

  int expa[def_params->fma + 1];
  int expb[def_params->fma + 1];
  int expprod[def_params->fma + 1];

  a[def_params->fma] = c[0];
  frexp(a[def_params->fma], &expprod[def_params->fma]);
  expprod[def_params->fma]++;
  int maxexp = expprod[def_params->fma];

  int i;
  /* Compute element-wise product of a and b.
     Get exponents of a, b, and of the products.
     Find the maximum product exponent.
  */
  for (i = 0; i < def_params->fma; i++) {
    frexp(a[i], &expa[i]);
    expa[i]++;
    frexp(b[i], &expb[i]);
    expb[i]++;
    a[i] = a[i] * b[i];
    frexp(a[i], &expprod[i]);
    expprod[i]++;
    if (expprod[i] > maxexp)
      maxexp = expprod[i];
  }

  /* Mask off significand bits to simulate
     significand alignment step of multi-term adders.
  */
  for (i = 0; i <= def_params->fma; i++) {
    uint64_t temp = d2u(a[i]);
    if ((i == def_params->fma) || expprod[i] == (expa[i] + expb[i]))
      temp = temp & (alignment_mask_norm_prod << (maxexp - expprod[i]));
    else
      temp = temp & (alignment_mask_denorm_prod << (maxexp - expprod[i]));
    a[i] = u2d(temp);
  }

  /* Perform reduction exactly */
  c[0] = 0;
  for (i = 0; i <= def_params->fma; i++) {
    c[0] += a[i];
  }
}
