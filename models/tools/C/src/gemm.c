/*
 * gemm.c - MATLAB external interface to tensor core GEMM.
 *
 * Performs alpha*A*B+beta*C
 *
 * The calling syntax is:
 *
 *		outMatrix = gemm(alpha, A, B, beta, C, informat, outformat, def_formats)
 *
 * This is a MEX file for MATLAB.
 */

#include "mex.h"
#include "../include/gemm.h"
#include <string.h>

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  /* Do various checks of the input args */
  if(nrhs != 8) {
    mexErrMsgIdAndTxt("MyToolbox:GEMM:nrhs",
                      "Eight inputs required.");
  }

  if(nlhs != 1) {
    mexErrMsgIdAndTxt("MyToolbox:GEMM:nlhs",
                      "One output required.");
  }

  /* Check alpha and beta are scalars */
  if( !mxIsDouble(prhs[0]) ||
      mxIsComplex(prhs[0]) ||
      mxGetNumberOfElements(prhs[0]) != 1 ) {
    mexErrMsgIdAndTxt("MyToolbox:GEMM:notScalar",
                      "Alpha must be a scalar.");
  }

  if( !mxIsDouble(prhs[3]) ||
      mxIsComplex(prhs[3]) ||
      mxGetNumberOfElements(prhs[3]) != 1 ) {
    mexErrMsgIdAndTxt("MyToolbox:GEMM:notScalar",
                      "Beta must be a scalar.");
  }

  if( !mxIsDouble(prhs[1]) ||
      mxIsComplex(prhs[1])) {
    mexErrMsgIdAndTxt("MyToolbox:GEMM:notDouble",
                      "A matrix must be type double.");
  }

  if( !mxIsDouble(prhs[2]) ||
      mxIsComplex(prhs[2])) {
    mexErrMsgIdAndTxt("MyToolbox:GEMM:notDouble",
                      "B matrix must be type double.");
  }

  if( !mxIsDouble(prhs[4]) ||
      mxIsComplex(prhs[4])) {
    mexErrMsgIdAndTxt("MyToolbox:GEMM:notDouble",
                      "C matrix must be type double.");
  }

  if( !mxIsChar(prhs[5])) {
    mexErrMsgIdAndTxt("MyToolbox:GEMM:notChar",
                      "informat must be a string");
  }

  if( !mxIsChar(prhs[6])) {
    mexErrMsgIdAndTxt("MyToolbox:GEMM:notChar",
                      "informat must be a string");
  }

  /* Check that input matrices are compatible for GEMM */
  if(mxGetN(prhs[1]) != mxGetM(prhs[2])) {
    mexErrMsgIdAndTxt("MyToolbox:GEMM:notRowVector",
                      "Number of column in A not equal number of rows in B");
  }

  if( !mxIsStruct(prhs[7])) {
    mexErrMsgIdAndTxt("MyToolbox:GEMM:notStruct",
                      "def_params must be a structure");
  }

  double alpha;
  double *A;
  double *B;
  double beta;
  double *C;
  char informat[20];
  char outformat[20];
  tc_params *def_params;
  def_params = malloc(sizeof(tc_params));

  int m, k, n;

  alpha = mxGetScalar(prhs[0]);
  beta = mxGetScalar(prhs[3]);

  A = mxGetDoubles(prhs[1]);
  B = mxGetDoubles(prhs[2]);
  C = mxGetDoubles(prhs[4]);

  mxGetString(prhs[5], informat, 20);
  mxGetString(prhs[6], outformat, 20);

  mxArray *temp = mxGetField(prhs[7], 0, "fma");
  def_params -> fma = (int)mxGetScalar(temp);
  temp = mxGetField(prhs[7], 0, "neab");
  def_params -> neab = (int)mxGetScalar(temp);
  temp = mxGetField(prhs[7], 0, "frmode");
  mxGetString(temp, def_params -> frmode, 4);
  temp = mxGetField(prhs[7], 0, "stkbitenabled");
  def_params -> stkbitenabled = (bool)mxGetScalar(temp);
  temp = mxGetField(prhs[7], 0, "inter_pattern");
  def_params -> inter_pattern = (bool)mxGetScalar(temp);

  /* Get dimensions of the input matrices */
  m = mxGetM(prhs[1]);
  k = mxGetN(prhs[1]);
  n = mxGetN(prhs[2]);

  /* If k is shorter than fma length, reduce fma length to k */
  if (k < def_params->fma)
    def_params->fma = k;

  /* Create the output matrix */
  plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
  double *outMatrix = mxGetDoubles(plhs[0]);

  /* Call the active gemm implementation */
  gemm_run(alpha, A, B, beta, C, informat, outformat, def_params, m, k, n);

  memcpy(outMatrix, C, sizeof(double) * m * n);
  free(def_params);
}
