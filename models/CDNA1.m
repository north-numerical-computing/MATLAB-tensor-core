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

%--------------------------------------------------------------------------
% Assumes FP16 inputs with FP32 accumulation/output precision.
% See Generic_TC_Model.m for full implementation details.
%==========================================================================

%---------------- Core configuration ----------------%
def_params.fma  = 4;        % Number of products in one FMA group
def_params.neab = Inf;        % Number of extra alignment bits (guard precision)

%---------------- Rounding configuration ----------------%
def_params.frmode         = 'rne';  % Final rounding mode:
                                   % 'rne' = round-to-nearest-even
def_params.armode = 'rd';   % Rounding mode during 2-operand alignment:
                                   % 'rd' = round-down (towards -Inf)
                                   % (multi-operand alignment uses truncation)

def_params.stkbitenabled  = 0;      % Enable sticky bit during alignment (1 = enabled)

%---------------- Accumulation architecture ----------------%
def_params.global_alignment  = 0;   % Align all products (and optionally c) to a common exponent
def_params.late_partial_sum  = 0;   % Add accumulation term 'c' after product summation
                                   % (products kept in denormalised form)
def_params.odd_even_grouping = 0;   % Enable separate accumulation of odd/even उत्पाद
def_params.pair_wise_sum     = 0;   % Enable pair-wise summation (not implemented)
def_params.denorm_prd        = 0;  % 1 keep denormalised, 0 normalised

%---------------- Exponent handling ----------------%
def_params.min_exp_limit   = -1024; % Minimum exponent allowed for product alignment
def_params.c_min_exp_limit = 0;     % Control minimum exponent for c:
                                   % 1 → clamp to -126 (FP32 subnormal boundary)
                                   % 0 → allow special handling when c = 0
def_params.prd_limit = 0;          % products are limited by output exponent bits, 1: limited, 0: allowed to exceed

%---------------- Accuracy / reference model ----------------%
def_params.correct_rounding = 1;    % Enable exact (Kulisch-style) accumulation
                                   % (used as reference / ground truth model)
                                
%---------------- Subnormal handling ----------------%
def_params.in_subnormals  = 1;   % Input subnormal support:
                                % 1 → preserve and process subnormals
                                % 0 → flush subnormals to zero (FTZ)
def_params.out_subnormals = 1;   % Output subnormal support:
                                % 1 → generate subnormal outputs
                                % 0 → flush subnormal results to zero



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
