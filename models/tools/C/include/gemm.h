#ifndef GEMM_H
#define GEMM_H

#include <stdbool.h>
#include <stdint.h>

/* Defines the data structure for setting up the tensor core parameters */
typedef struct {
  int fma;
  int neab;
  char frmode[4]; // Three chars for the rounding mode + terminator '\0'.
  bool stkbitenabled;
  bool inter_pattern;
} tc_params;

extern int accum_prec;
extern uint64_t alignment_mask_norm_prod;
extern uint64_t alignment_mask_denorm_prod;

/* Signature for gemm implementations */
typedef void (*gemm_alg_t)(double alpha, double *A, double *B, double beta,
                           double *C, char *informat, char *outformat,
                           tc_params *def_params, int m, int k, int n);

/* NVIDIA-like multi-term adder of products with no intermediate normalisation */
void block_FMA_nv(double *a, double *b, double *c, tc_params *def_params);

/* Set which gemm implementation is active */
void gemm_set(gemm_alg_t gemm_selection);

/* Pick up the current GEMM alg. and run it */
void gemm_run(double alpha, double *A, double *B, double beta, double *C,
              char *informat, char *outformat, tc_params *def_params,
              int m, int k, int n);

/* Various gemm algorithms */

void gemm_recur(double alpha, double *A, double *B, double beta, double *C,
                char *informat, char *outformat, tc_params *def_params,
                int m, int k, int n);

#endif
