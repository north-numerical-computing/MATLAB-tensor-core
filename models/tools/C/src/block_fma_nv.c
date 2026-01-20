#include "../include/gemm.h"
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "mex.h"

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

   Inputs a, b, and c are assumed to be exact.

   Elementwise product a*b is assumed to be kept exactly.
 */
void block_FMA_nv(double *a, double *b, double *c, tc_params *def_params) {

  int expa[def_params->fma + 1];
  int expb[def_params->fma + 1];
  int expprod[def_params->fma + 1];

  int largest_magn_index = 0;
  int inc_prec = 0;

  a[def_params->fma] = c[0];
  frexp(a[def_params->fma], &expprod[def_params->fma]);
  if (c[0])
    expprod[def_params->fma]--;

  int maxexp;
  if (c[0]) {
    maxexp = expprod[def_params->fma];
    largest_magn_index = def_params->fma;
  }
  else
    maxexp = -1074;

  int i;
  /* Compute element-wise product of a and b.
     Get exponents of a, b, and of the products.
     Find the index of the product with the max exponent.
  */
  for (i = 0; i < def_params->fma; i++) {
    frexp(a[i], &expa[i]);
    if (a[i])
      expa[i]--;
    frexp(b[i], &expb[i]);
    if (b[i])
      expb[i]--;
    a[i] = a[i] * b[i];
    frexp(a[i], &expprod[i]);
    if (a[i])
      expprod[i]--;

    // Find largest exponent within (maybe denormalised) product exponents.
    int denorm_exp = expprod[i];
    if (expprod[i] != expa[i] + expb[i])
      denorm_exp--;
    if (a[i] && denorm_exp >= maxexp) {
      int temp = largest_magn_index;
      largest_magn_index = i;
      if (denorm_exp == maxexp && expprod[i] == expa[i] + expb[i])
        largest_magn_index = temp;
      maxexp = denorm_exp;
    }
  }

  // Increase precision by 1 if the largest exponent is associated
  // with a product that has a 2nd bit to the left of binary point set.
  if (largest_magn_index != def_params->fma &&
      expprod[largest_magn_index] !=
      (expa[largest_magn_index] + expb[largest_magn_index])) {
    inc_prec = 1;
  }

  // Return to normalised state.
  maxexp = expprod[largest_magn_index];

  /* Mask off significand bits to simulate
     significand alignment step of multi-term adders.
  */
  for (i = 0; i <= def_params->fma; i++) {
    uint64_t temp = d2u(a[i]);
    if (inc_prec) {
      temp = temp & (alignment_mask_denorm_prod << (maxexp - expprod[i]));
    } else {
      temp = temp & (alignment_mask_norm_prod << (maxexp - expprod[i]));
    }
    // If the alignment is so large that the implicit bit drops off,
    // make the product zero.
    if ((maxexp - expprod[i]) > accum_prec + inc_prec)
      temp = 0;
    a[i] = u2d(temp);
  }

  /* Perform reduction exactly */
  c[0] = 0;
  for (i = 0; i <= def_params->fma; i++) {
    c[0] += a[i];
  }
}
