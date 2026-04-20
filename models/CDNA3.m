function D=CDNA3(alpha, A, B, beta, C, informat)
%
% CDNA3 Architecture: Compute GEMM using a model of Matrix Cores (MC)
% for AMD MI300/MI300A/MI300X and other GPUs with the same CDNA3 architecture.
%
% This function evaluates the expression:
%       D = alpha * A * B + beta * C
% using a numerical-feature-based model of CDNA3 Matrix Cores.
% The accumulation of partial block products is performed using
% recursive summation.
%
% Inputs:
%   alpha    : Scalar multiplier for A * B.
%   A        : Left matrix operand.
%   B        : Right matrix operand.
%   beta     : Scalar multiplier for C.
%   C        : Matrix added to the product.
%   informat : String specifying the numerical format of A and B.
%              Supported input formats:
%                  'fp16', 'binary16', 'half',
%                  'bf16', 'bfloat16', 'brainfloat16',
%                  'tf32', 'tensorfloat32', 'xf32',
%                  'fp8', 'bf8', 'fp8-e5m2', 'fp8-e4m3'
%              Note:
%                  fp32/fp64 correspond to sequential FMAs and are not
%                  implemented in this model (can be computed directly in MATLAB).
%
% Output:
%   D : Result of the operation D = alpha * A * B + beta * C computed
%       under the specified CDNA3 tensor core configuration.
%

% Allowed formats
allowedInFormats = {'fp16','binary16', 'half','bf16','bfloat16','tf32','tensorfloat32','xf32','fp8-e5m2','fp8-e4m3','fp8','bf8'};

if exist('informat', 'var')
    if (~ismember(lower(informat), allowedInFormats))
        error('The specified input format is not supported.');
    end
    informat=lower(informat);
end

% Default parameters assuming fp16 input and fp32 accumulation/output.
% See Generic_TC_Model.m for further details.
def_params.fma = 8;          % Fused multiply-add (FMA) group size
def_params.neab = 1;         % Extra alignment bits
def_params.frmode = 'rne';  % Final rounding mode (round-to-nearest-even)
def_params.model='CDNA3';   % GEMM is changed to require model to run CDNA 3

% Configure model based on input format
if ismember(informat, {'fp16','half','binary16','bf16','brainfloat16'})
   % Default configuration

elseif ismember(informat, {'tf32','tensorfloat32','xf32'}) 
        def_params.fma = 4;   

elseif ismember(informat, {'bf8','fp8','fp8-e5m2','fp8-e4m3','e5m2','e4m3'}) 
        def_params.fma = 16;   

else
  % Should not be reached due to earlier validation
  error("Error detected in informat setting in CDNA3.m file");
end
    
D = GEMM(alpha, A, B, beta, C, informat, 'binary32', def_params);
        
end
