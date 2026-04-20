function D=CDNA1(alpha, A, B, beta, C, informat)
%
% CDNA1 Architecture: Compute GEMM using a model of Matrix Cores (MC)
% for AMD MI100 and other GPUs with the same CDNA1 architecture.
%
% This function evaluates the expression:
%       D = alpha * A * B + beta * C
% using a numerical-feature-based model of CDNA1 Matrix Cores.
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
%                  'bf16', 'bfloat16', 'brainfloat16'
%              Note:
%                  fp32/fp64 correspond to sequential FMAs and are not
%                  implemented in this model (can be computed directly in MATLAB).
%
% Output:
%   D : Result of the operation D = alpha * A * B + beta * C computed
%       under the specified CDNA1 tensor core configuration.
%

% Allowed formats
allowedInFormats = {'fp16','binary16', 'half','bf16','bfloat16','brainfloat16'};

if exist('informat', 'var')
    if (~ismember(lower(informat), allowedInFormats))
        error('The specified input format is not supported.');
    end
    informat=lower(informat);
end

% Default parameters assuming fp16 input and fp32 accumulation/output.
% See Generic_TC_Model.m for more details.
def_params.fma = 4;          % Fused multiply-add (FMA) group size
def_params.neab = Inf;       % Extra alignment bits
def_params.frmode = 'rne';  % Final rounding mode (round-to-nearest-even)
def_params.model='CDNA1';

% Configure model based on input format
if ismember(informat, {'fp16','half','binary16'})
    if exist('outformat', 'var')
        if ismember(outformat, {'fp16','binary16','half'})
            def_params.frmode='rne'; % Final rounding mode
        end  
    end

elseif ismember(informat, {'bf16','brainfloat16','bfloat16'}) 
        def_params.fma = 2; 
else
    % Should not be reached due to earlier validation
    error('Input format is not supported')
end

D = GEMM(alpha, A, B, beta, C, informat, 'binary32', def_params);

end
