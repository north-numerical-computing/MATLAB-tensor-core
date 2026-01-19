#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../include/gemm.h"
#include "cpfloat_binary64.h"

/* Set some tensor core masks */
int accum_prec;
uint64_t alignment_mask_norm_prod;
uint64_t alignment_mask_denorm_prod;


void gemm_recur(double alpha, double *A, double *B, double beta, double *C,
                char *informat, char *outformat, tc_params *def_params, int m,
                int k, int n) {

  /* Round a and b to the input format */
  optstruct *informat_opts = init_optstruct();
  strcpy(informat_opts->format, informat);
  cpfloat_populate_optstruct_from_format(informat_opts);
  cpfloat(A, A, m*k, informat_opts);
  cpfloat(B, B, k * n, informat_opts);

  /* Round c to the output format*/
  optstruct *outformat_opts = init_optstruct();
  strcpy(outformat_opts->format, outformat);
  cpfloat_populate_optstruct_from_format(outformat_opts);
  cpfloat(C, C, m * n, outformat_opts);

  /* Set some tensor core masks */
  accum_prec = outformat_opts->precision - 1 + def_params->neab;
  alignment_mask_norm_prod =
    UINT64_MAX << (52-accum_prec);
  alignment_mask_denorm_prod =
    UINT64_MAX << (52 - accum_prec - 1);

  if (strcmp(def_params->frmode, "rz")==0)
    outformat_opts->round = CPFLOAT_RND_TZ;

  int i, j, r, l;

  /* Scale A by alpha */
  for (i = 0; i < m; i++)
    for (r = 0; r < k; r++)
      A[r * m + i] = alpha * A[r * m + i];

  /* Scale C by beta */
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      C[j * m + i] = beta * C[j * m + i];


  #pragma omp parallel default(none) \
    shared(A, B, C, m, n, k, def_params, outformat_opts) private(i, j, r, l)
  {
    /* Allocate small temporary storage for tensor core inputs.
       +1 element in a for storing the elementwise products and c*/
    double *a = (double*) malloc((def_params->fma+1) * sizeof *a);
    double *b = (double*) malloc(def_params->fma * sizeof *b);

    /* Compute GEMM */
    #pragma omp for collapse(2)
    for (i = 0; i < m; i++)
      for (j = 0; j < n; j++) {
        for (r = 0; r < k / def_params->fma; r++) {
          for (l = 0; l < def_params->fma; l++) {
            a[l] = A[(r * def_params->fma + l) * m + i];
            b[l] = B[j * k + r * def_params->fma + l];
          }
          block_FMA_nv(a, b, &C[j * m + i], def_params);
          cpfloat(&C[j * m + i], &C[j * m + i], 1, outformat_opts);
        }
      }

    free(a);
    free(b);
  }

  /* Perform final rounding */
  //cpfloat(C, C, m * n, outformat_opts);
}
