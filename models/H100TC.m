function D=H100TC(alpha, A, B, beta, C, informat, outformat)
%
% H100TC  Compute GEMM with a model of a tensor core of the H100 GPU.
%
% This function evaluates the expression D = A * B + C using the 
% H200 TC numerical-feature-based model. The accumulation of block
% products is performed using recursive summation.
%
% Inputs
%   A: Left matrix operand for the matrix multiplication A * B.
%   B: Right matrix operand for the matrix multiplication A * B.
%   C: Matrix added to the product A * B.
%   informat: a string specifying the format of A and B.
%             Supported input formats:
%                  fp8-(e5m2,e4m3), fp16, binary16, half,
%                  bf16, bfloat16, tensorfloat32, tf32.
%   outformat: a string specifying the numerical format for C and D.
%              Supported output formats:
%                  fp32, single, binary32,
%                  fp16, binary16, half.
%
% Output
%   D: Result of the operation D = A * B + C computed under the 
%      specified tensor core configuration.

D = B200TC(alpha, A, B, beta, C, informat, outformat);
