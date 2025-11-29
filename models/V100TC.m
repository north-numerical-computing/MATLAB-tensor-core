function D = V100TC(alpha, A, B, beta, C, outformat)
%
% V100TC  Compute GEMM with a model of a tensor core of the V100 GPU.
%
% This function evaluates the expression D = A * B + C using the 
% V100 TC numerical-feature-based model. The accumulation of block
% products is performed using recursive summation.
%
% Inputs
%   A: Left matrix operand for the matrix multiplication A * B.
%   B: Right matrix operand for the matrix multiplication A * B.
%   C: Matrix added to the product A * B.
%   outformat: a string specifying the numerical format for C and D.
%              Supported output formats:
%                  fp32, single, binary32,
%                  fp16, binary16, half.
%
% Output
%   D: Result of the operation D = A * B + C computed under the 
%      specified tensor core configuration.

% Allowed formats
allowedOutFormats = {'fp32', 'single', 'binary32',...
    'fp16', 'binary16', 'half'};

if (exist('outformat', 'var'))
    if (~ismember(lower(outformat), allowedOutFormats))
        error('The specified output format is not supported.');
    end
    outformat=lower(outformat);
end

% Default structures assuming fp16 in and fp32 output. See
% Generic_TC_Model.m for the information.
def_params.fma = 4;          % Fused multiply-add (FMA) size
def_params.neab = 0;          % TC extra alignment bits
def_params.frmode = 'rz';     % TC final rounding mode
def_params.stkbitenabled = 0;
def_params.inter_pattern=0;

if nargin > 3
    if ismember(outformat,{'fp16','binary16','half'})
        def_params.frmode='rne';
    end
end

D = GEMM(alpha, A, B, beta, C, "binary16", outformat, def_params);

end
































































