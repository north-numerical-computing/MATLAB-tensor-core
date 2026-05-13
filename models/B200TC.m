function D = B200TC(alpha, A, B, beta, C, informat, outformat)
%
% B200TC  Compute GEMM with a model of a tensor core of the B200 GPU.
%
% This function evaluates the expression D = A * B + C using the 
% B200 TC numerical-feature-based model. The accumulation of block
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

%addpath('tools')

% Allowed formats
allowedOutFormats = {'fp32', 'single', 'binary32',...
    'fp16', 'binary16', 'half'};
allowedInFormats = {'fp8-e5m2','fp8-e4m3','e5m2','e4m3',...
    'fp16','binary16', 'half','bf16','bfloat16','tensorfloat32','tf32'};

if exist('informat', 'var')
    if (~ismember(lower(informat), allowedInFormats))
        error('The specified input format is not supported.');
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
%---------------- Core configuration ----------------%
def_params.fma  = 16;        % Number of products in one FMA group
def_params.neab = 2;        % Number of extra alignment bits (guard precision)

%---------------- Rounding configuration ----------------%
def_params.frmode = 'rz';  % Final rounding mode:
                                   % 'rne' = round-to-nearest-even
def_params.armode = 'rz';   % Rounding mode during 2-operand alignment:
                                   % 'rd' = round-down (towards -Inf)
                                   % (multi-operand alignment uses truncation)

def_params.stkbitenabled  = 0;      % Enable sticky bit during alignment (1 = enabled)

%---------------- Accumulation architecture ----------------%
def_params.global_alignment  = 1;   % Align all products (and optionally c) to a common exponent
def_params.late_partial_sum  = 0;   % Add accumulation term 'c' after product summation
                                   % (products kept in denormalised form)
def_params.odd_even_grouping = 0;   % Enable separate accumulation of odd/even उत्पाद
def_params.pair_wise_sum     = 0;   % Enable pair-wise summation (not implemented)
def_params.denorm_prd        = 1;  % 1 keep denormalised, 0 normalised

%---------------- Exponent handling ----------------%
def_params.min_exp_limit   = -133; % Minimum exponent allowed for product alignment
def_params.c_min_exp_limit = 0;     % Control minimum exponent for c:
                                   % 1 → clamp to -126 (FP32 subnormal boundary)
                                   % 0 → allow special handling when c = 0
def_params.prd_limit = 0;          % products are limited by output exponent bits, 1: limited, 0: allowed to exceed

%---------------- Accuracy / reference model ----------------%
def_params.correct_rounding = 0;    % Enable exact (Kulisch-style) accumulation
                                   % (used as reference / ground truth model)
                                
%---------------- Subnormal handling ----------------%
def_params.in_subnormals  = 1;   % Input subnormal support:
                                % 1 → preserve and process subnormals
                                % 0 → flush subnormals to zero (FTZ)
def_params.out_subnormals = 1;   % Output subnormal support:
                                % 1 → generate subnormal outputs
                                % 0 → flush subnormal results to zero


% Set up the model according to the formats specified.
if ismember(informat, {'fp16','half','binary16'})
    if exist('outformat', 'var')
        if ismember(outformat, {'fp16','binary16','half'})
            def_params.frmode='rne'; % TC final rounding mode
        end  
    end
elseif ismember(informat, {'tf32', 'tensorfloat32'})
         def_params.fma=8;
elseif ismember(informat, {'fp8-e5m2','fp8-e4m3','e5m2','e4m3'}) 
        % FMA size is 16, but interleaved pattern is used to join two
        % 16-element vectors.
        def_params.fma = 32;   
        if exist('outformat', 'var')
            outformat='fp32';
            if ismember(outformat, {'fp16','binary16','half'})
                def_params.frmode='rne';
            end
        end
end

     D = GEMM(alpha, A, B, beta, C, informat, outformat, def_params);
        
end































































