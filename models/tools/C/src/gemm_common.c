#include <stdio.h>
#include "../include/gemm.h"

static void default_gemm(double alpha, double *A, double *B, double beta,
                         double *C, char *informat, char *outformat,
                         tc_params *def_params, int m, int k, int n) {
  gemm_recur(alpha, A, B, beta, C, informat, outformat, def_params, m, k, n);
}

static gemm_alg_t active_gemm = default_gemm;

/* Set which gemm implementation is active */
void gemm_set(gemm_alg_t gemm_selection) {
  active_gemm = gemm_selection;
}

/* Pick up the current GEMM alg. and run it */
void gemm_run(double alpha, double *A, double *B, double beta, double *C,
              char *informat, char *outformat, tc_params *def_params, int m,
              int k, int n) {
  active_gemm(alpha, A, B, beta, C, informat, outformat, def_params, m, k, n);
}
