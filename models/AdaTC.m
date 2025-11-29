function D = AdaTC(alpha, A, B, beta, C, informat, outformat)
%
% AdaTC  Compute GEMM with a model of a tensor core of the Ada RTX 1000
% GPU.
%
% This function evaluates the expression D = A * B + C using the 
% Ada RTX 1000 TC numerical-feature-based model. The accumulation of block
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


% Allowed formats
allowedOutFormats = {'fp32', 'single', 'binary32',...
    'fp16', 'binary16', 'half'};
allowedInFormats = {'fp8-e5m2','fp8-e4m3','e5m2','e4m3',...
    'fp16','binary16', 'half','bf16','bfloat16','tensorfloat32','tf32'};

if exist('informat', 'var')
    if (~ismember(lower(informat), allowedInFormats))
        error('The specified input format not supported.');
    end
    informat=lower(informat);
end
if (exist('outformat', 'var'))
    if (~ismember(lower(outformat), allowedOutFormats))
        error('The specified output format is not supported.');
    end
    outformat=lower(outformat);
end

% Default structures assuming fp16 in and fp32 output. See
% Generic_TC_Model.m for the information.
def_params.fma = 8;          % Fused multiply-add (FMA) size
def_params.neab = 1;          % TC extra alignment bits
def_params.frmode = 'rz';     % TC final rounding mode
def_params.stkbitenabled = 0;
def_params.inter_pattern=0;

% Set up the model according to the formats specified.
if ismember(informat, {'fp16','half','binary16'})
    if exist('outformat', 'var')
        if ismember(outformat, {'fp16','binary16','half'})
            def_params.frmode='rne';
        end
    end
elseif ismember(informat, {'tf32', 'tensorfloat32'})
         def_params.fma=4;
elseif ismember(informat, {'fp8-e5m2','fp8-e4m3','e5m2','e4m3'}) 
        def_params.fma = 16;
        def_params.inter_pattern=0;
        def_params.neab=-10;
        
        if exist('outformat', 'var')
            if ismember(outformat, {'fp16','binary16','half'})
                def_params.frmode='rne';
            end
        end
end

D = GEMM(alpha, A, B, beta, C, informat, outformat, def_params);

end
































































