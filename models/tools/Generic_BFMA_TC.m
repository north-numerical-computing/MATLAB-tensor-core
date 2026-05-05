function d = Generic_BFMA_TC(a_block, b_block, c, model_params)
%==========================================================================
% Generic_BFMA_TC
%--------------------------------------------------------------------------
% Simulates Block Floating Multiply-Accumulate (BFMA) with configurable
% architectural behaviors (e.g., CDNA, NVIDIA-like models).
%
% Inputs:
%   a_block, b_block : input vectors (multiplicands)
%   c                : accumulation term
%   model_params     : structure containing model configuration
%
% Output:
%   d : final floating-point result
%==========================================================================


%% =========================================================================
% Extract format parameters
% =========================================================================
%NoManBitsIn   = model_params.NoManBitsIn;
NoManBitsOut  = model_params.NoManBitsOut;
NoExpBitsIn   = model_params.NoExpBitsIn;
NoExpBitsOut  = model_params.NoExpBitsOut;

%% =========================================================================
% Extract model configuration flags
% =========================================================================
stkbitenabled     = model_params.stkbitenabled;
OutRoundMode      = model_params.frmode;
neab              = model_params.neab;
correct_rounding  = model_params.correct_rounding;
global_alignment  = model_params.global_alignment;
late_partial_sum  = model_params.late_partial_sum;
odd_even_grouping = model_params.odd_even_grouping;
pair_wise_sum     = model_params.pair_wise_sum;
min_exp_limit     = model_params.min_exp_limit;
c_min_exp_limit   = model_params.c_min_exp_limit;
align_round_mode  = model_params.armode;
out_subnormals     = model_params.out_subnormals;
in_subnormals     = model_params.in_subnormals;
prd_limit         = model_params.prd_limit;
%% =========================================================================
% Initialization
% =========================================================================
emin_output = 1 - (2^(NoExpBitsOut - 1) - 1);
emin_input  = 1 - (2^(NoExpBitsIn  - 1) - 1);

% remove subnormals in input
if ~in_subnormals
    a_block(abs(a_block)<2^emin_input)=0;
    b_block(abs(b_block)<2^emin_input)=0;
end
% make c=0, if its subnormal 
if ~out_subnormals & abs(c)<(2^emin_output)
    c=0;
end
% Identify positions where product will be zero
prod_zeroIdxs = (a_block == 0) | (b_block == 0);

% Remove zero-valued products (unless grouping requires them)
if ~odd_even_grouping && ~pair_wise_sum
    a_block(prod_zeroIdxs) = [];
    b_block(prod_zeroIdxs) = [];
end

% Early exit: all products and accumulation are zero
c_zero_check = (c == 0);
if isempty(a_block) && c_zero_check
    d = 0;
    return
end

% check if products are allowed to exceed 2^NoExpBitsOut 
if prd_limit
    prd = a_block .* b_block;
    prod_max_limit = 2^NoExpBitsOut;

    if any(prd >= prod_max_limit) || any(prd <= -prod_max_limit)
        if any(prd >= prod_max_limit) && any(prd <= -prod_max_limit)
            d = NaN;   % overflow on both positive and negative sides
        else
            d = Inf;   % overflow only on one side
        end
        return
    end
end

%% =========================================================================
% Compute exponents and significands of inputs
% =========================================================================
if ~correct_rounding && (~pair_wise_sum)
    a_block_abs = abs(a_block);
    b_block_abs = abs(b_block);
    
    [~, a_exp] = log2(a_block_abs);
    [~, b_exp] = log2(b_block_abs);
    
    a_exp = max(a_exp - 1, emin_input);
    b_exp = max(b_exp - 1, emin_input);
    
    a_sig = pow2(a_block, -a_exp);
    b_sig = pow2(b_block, -b_exp);
    
    % Compute product representation
    prod_sig = a_sig .* b_sig;
    prod_exp = a_exp + b_exp;
    
    % Handle zero products in grouping mode
    if odd_even_grouping
        prod_exp(prod_zeroIdxs) = -1024; % avoid -Inf issues
    end
    
    % Extract sign bits
    sign_bits = (prod_sig < 0);

end

%% =========================================================================
% ===================== MODEL SELECTION ====================================
% =========================================================================

%--------------------------------------------------------------------------
% Model 1: Exact accumulation (Kulisch accumulator)
%--------------------------------------------------------------------------
if correct_rounding
    
    if ~c_zero_check
        sign_bits(end+1) = (c < 0);
    end
    
    [dbits, dexp, sOut] = Kulisch_Accumulation( ...
        prod_sig, prod_exp, c, sign_bits, c_zero_check);


%--------------------------------------------------------------------------
% Pair-wise summation (not yet implemented)
%--------------------------------------------------------------------------
elseif pair_wise_sum
    % this function does not require product significand and exponent
    % separetely
    d = pws_main(a_block,b_block,c,in_subnormals,out_subnormals,OutRoundMode,emin_input,emin_output);
    return;
    
    % TODO: Implement pair-wise accumulation


%--------------------------------------------------------------------------
% Global alignment model (e.g., NVIDIA / CDNA-3)
%--------------------------------------------------------------------------
elseif global_alignment
    
    % Include accumulation term early unless delayed
    if ~c_zero_check && ~late_partial_sum
        sign_bits(end+1) = (c < 0);
    end
    
    [dbits, dexp, sOut, prod_max_exp, prod_sum] = ...
        GlobalAlignmentSum(prod_sig, prod_exp, ...
        c * (~late_partial_sum), sign_bits, ...
        neab, stkbitenabled, min_exp_limit, 0);
    
    
    %---------------- Late partial sum ----------------
    if late_partial_sum
        
        % Normalize accumulation term c
        if ~c_zero_check
            [~, c_exp] = log2(abs(c));
            c_exp = max(c_exp - 1, -126);
            c_sig = c / 2^c_exp;
            
            c_sig_uint = uint32(abs(c_sig) * 2^(23 + neab));
            c_sign = (c < 0);
        else
            c_sig_uint = 0;
            c_sign = 0;
            c_exp = -126 * c_min_exp_limit - 1024 * (~c_min_exp_limit);
        end
        
        c_exp = int16(c_exp);
        
        % Add partial sums
        [max_exp, prod_sum_unnorm, sOut, neab] = ...
            AddTwoNonNormSums(prod_max_exp, c_exp, sOut, ...
            c_sign, prod_sum, c_sig_uint, ...
            align_round_mode, 31-24, 24-24, neab);
        
        % Normalize result
        [dbits, dexp] = norm_helper(prod_sum_unnorm, max_exp, neab, 0);
    end


%--------------------------------------------------------------------------
% Odd-even grouping model (e.g., CDNA-3 FP8 behavior)
%--------------------------------------------------------------------------
elseif odd_even_grouping
    
    % Even-indexed products
    [~, ~, even_sign, even_exp, even_sum] = ...
        GlobalAlignmentSum(prod_sig(1:2:end), prod_exp(1:2:end), ...
        0, sign_bits(1:2:end), neab, 0, min_exp_limit, 0);
    
    % Odd-indexed products
    [~, ~, odd_sign, odd_exp, odd_sum] = ...
        GlobalAlignmentSum(prod_sig(2:2:end), prod_exp(2:2:end), ...
        0, sign_bits(2:2:end), neab, 0, min_exp_limit, 0);
    
    % Combine even and odd sums
    [max_exp, prod_sum, sOut, neab] = ...
        AddTwoNonNormSums(even_exp, odd_exp, even_sign, ...
        odd_sign, even_sum, odd_sum, ...
        align_round_mode, 0, 0, neab);
    
    % Add accumulation term c
    if ~c_zero_check
        [~, c_exp] = log2(abs(c));
        c_exp = max(c_exp - 1, -126);
        c_sig = c / 2^c_exp;
        
        c_sig_uint = uint32(abs(c_sig) * 2^24);
        c_sign = (c < 0);
    else
        c_sig_uint = 0;
        c_sign = 0;
        c_exp = int16(-126 * c_min_exp_limit - 1024 * (~c_min_exp_limit));
    end
    
    [max_exp, prod_sum_unnorm, sOut, neab] = ...
        AddTwoNonNormSums(max_exp, c_exp, sOut, ...
        c_sign, prod_sum, c_sig_uint, ...
        align_round_mode, 31-24, 24-24, neab);
    
    [dbits, dexp] = norm_helper(prod_sum_unnorm, max_exp, neab, 0);


%--------------------------------------------------------------------------
% Placeholder for additional models
%--------------------------------------------------------------------------
else
    % Additional architectures can be implemented here
end


%% =========================================================================
% Handle subnormal numbers (before rounding)
% =========================================================================
if dexp < emin_output
    shift = dexp - emin_output;
    dbits = subnormalsignificand(dbits, abs(shift), 0);
    dexp  = emin_output;
end


%% =========================================================================
% Rounding
% =========================================================================
if ~strcmp(OutRoundMode, 'rz')
    [dbits, dexp] = ieeeround(dbits, OutRoundMode, ...
                             NoManBitsOut, sOut, double(dexp));
end


%% =========================================================================
% Handle subnormal numbers (after rounding)
% =========================================================================
if dexp < emin_output
    shift = dexp - emin_output;
    dbits = subnormalsignificand(dbits(1:2+NoManBitsOut), abs(shift), 1);
    dexp  = emin_output;
end

if isempty(dexp)
    dexp = 0;
end


%% =========================================================================
% Convert bit representation to decimal value
% =========================================================================
mantissa = (dbits(1) - '0') + ...
           bin2dec(dbits(3:NoManBitsOut+2)) * 2^(-NoManBitsOut);

d = mantissa * 2^double(dexp);

if sOut == 1
    d = -d;
end


%% =========================================================================
% Encode exponent with bias
% =========================================================================
dexp = dexp + (2^(NoExpBitsOut - 1) - 1);


%% =========================================================================
% Handle overflow (Inf / -Inf)
% =========================================================================
if dexp == (2^NoExpBitsOut - 1)
    d = Inf;
    if sOut == 1
        d = -d;
    end
    return
end


%% =========================================================================
% Final safety check
% =========================================================================
if isempty(d)
    d = 0;
end

% flush subnormal based on out_subnormal flag
if abs(d)<2^emin_output && ~out_subnormals
d=0;
end

end



%#######################################################################
% Pair Wise Sum Main Function: Implementing CDNA 2 Like Models
%#######################################################################
function d=pws_main(a,b,c,in_subnormal,out_subnormals,OutRoundMode,emin_input,emin_output)
        min_in=2^emin_input;
        min_out=2^emin_output;
        % flush out input subnormal
        if ~in_subnormal
            a_sn_idx=abs(a)<min_in;
            b_sn_idx=abs(b)<min_in;
            zero_idx=a_sn_idx | b_sn_idx;
            a(zero_idx)=0; % flushing a
            b(zero_idx)=0; % flushing b
        end
        prod=single(a.*b); % by default 
        % flush the out_subnormals
        if ~out_subnormals    
            c=(abs(c)>=min_out)*c;
            prod(abs(prod)<min_out)=0;
        end

    % recursive addition calling    
    d=c;     
    if strcmp(OutRoundMode,'rne')
        psout=pws_fma(prod,out_subnormals,min_out);
        d=fma(d,1,psout); % n/2:n-1
        if ~out_subnormals
            d=d*(abs(d)>=min_out);
        end
    else
        psout=pws_recursive_helper(prod,out_subnormals,emin_output,min_out,OutRoundMode);
        d=fma2(d,psout,emin_output,OutRoundMode); % n/2:n-1
        if ~out_subnormals
            d=d*(abs(d)>=min_out);
        end
        % other written function
    end



end

%% PairWise Recursive Helper Function
function d=pws_recursive_helper(prod,out_subnormals,emin_output,min_out,OutRoundMode)

if numel(prod)==1
    d=prod;
    return
else
    d=pws_recursive_helper(prod(1:end/2),out_subnormals,emin_output,min_out,OutRoundMode); % 0:n/2-1;
    d=fma2(d,pws_recursive_helper( prod(end/2+1:end),out_subnormals,emin_output,min_out,OutRoundMode),emin_output,OutRoundMode); % n/2:n-1
        if ~out_subnormals
            d=d*(abs(d)>=min_out);
        end
        if isnan(d)
            return
        end

        return;

end

end

%% Self coded fma function offering multiple rounding modes   
function d=fma2(d,prod,emin_output,OutRoundMode)
d=[d,prod];
[~,d_exp]=log2(abs(d));
d_exp=d_exp-1;
d_exp(d_exp<-126)=-126;
d_sig=pow2(d,-d_exp);
d_exp(d==0)=-1024;
% [maxExp, alignedSig] = AlignSignficand(d_sig,d_exp,0,2,1,-1024,0);
% sum=dot(double(alignedSig),(1-2*d<0));
% if sum==0
%     d=0;
%     return
% end
% sOut=sum<0;
%[dbits,dexp]=norm_helper(sum,maxExp,2,1);
[dbits, dexp, sOut] = Kulisch_Accumulation(d_sig, d_exp, 0,d<0,1);
% subnormal range
if dexp < emin_output
    shift = dexp - emin_output;
    dbits = subnormalsignificand(dbits, abs(shift), 0);
    dexp  = emin_output;
end
% rounding
if ~strcmp(OutRoundMode, 'rz')
    [dbits, dexp] = ieeeround(dbits, OutRoundMode,23, sOut, double(dexp));
end
%

mantissa = (dbits(1) - '0')+bin2dec(dbits(3:23+2)) * 2^(-23);
d = mantissa * 2^double(dexp);

if sOut == 1
    d = -d;
end

dexp = dexp + (2^(8 - 1) - 1);
if dexp == (2^8 - 1)
    d = Inf;
    if sOut == 1
        d = -d;
    end
    return
end



end

%% Pair-Wise Sum with matlab built-in fma function for rne rounding
function d=pws_fma(prod,out_subnormals,min_out)
    if numel(prod) == 1
        d=prod;
        return;
    else
        d=pws_fma(prod(1:end/2),out_subnormals,min_out); % 0:n/2-1
        d=fma(d,1,pws_fma( prod(end/2+1:end),out_subnormals,min_out)); % n/2:n-1
        if ~out_subnormals
            d=d*(abs(d)>=min_out);
        end
        if isnan(d)
            return
        end

        return;
    end
end


%==========================================================================
%% Function: 
% Purpose : Simulat the NVIDIA/AMD TC behaviour for a full FMA,
% Inputs
% - stkbitenabled : for stkbit in case of two operands to this function
% like only two products with c=0 or a single product with c non-zero,
% however, not used but was for inter_leaved pattern param for fp8 TC
% access via HMMA (fp16) TC where inter-leaved pattern is to be implemented
% and c is added at the end with RNE
%==========================================================================
%==========================================================================
function [dbits, dexp, signOut,max_exp,sum_unormalised] = GlobalAlignmentSum(prod_sig, prod_exp, c, sign_bits,neab,stkbitenabled,min_exp_limit,c_min_exp_limit)

    [max_exp, align_sigs] = AlignSignficand(prod_sig,prod_exp,c,neab,stkbitenabled,min_exp_limit,c_min_exp_limit);
    sum_unormalised=dot(double(align_sigs),(1-2*(sign_bits)));
    signOut=sum_unormalised<0;
    % if sum is 
     if sum_unormalised==0
            dexp=0;
            dbits=['0.00000000000000000000000'];
            return
     end
    [dbits,dexp]=norm_helper(sum_unormalised,max_exp,neab,stkbitenabled);
   
end

%==========================================================================
%% Denormalised Addition of two operands with different alignment mode
%==========================================================================
function [max_exp,prod_sum_unnorm,prod_sum_sign,neab]=AddTwoNonNormSums(max_exp_1,max_exp_2,sign_1,sign_2,sum_1,sum_2,align_round_mode,eab_1,eab_2,neab)
    
    max_exp=max([max_exp_1,max_exp_2]);
    shift=abs(max_exp_1-max_exp_2);
    
    if eab_1<0 || eab_2<0
        error('Alignment Bits Post Global or Grouped Global Alignment Should not be Negative');
    end


    % convert all sums to uint64 for safety
    sum_1=uint64(abs(sum_1)); sum_2=uint64(abs(sum_2));
    
    % shift>512 means one of the product sum is zero, and therefore, shift
    % must be zero
    if shift>532
        shift=0;
    end
    % unlike addition of c in this function and in CDNA3_High, we have an
    % extra elseif because zero condition is not separately checked
    ulpAdjument=0;
    if max_exp_1<max_exp   
        % round down even indexed product sum
        neab=neab+eab_1;
        sum_1=bitshift(sum_1,eab_1);
        sum_2=bitshift(sum_2,eab_1);
        switch align_round_mode
            case 'rd' 
                ulpAdjument=(sum_1-bitshift(bitshift(sum_1,-shift),shift))>0 && sign_1;
            case 'ru'
                ulpAdjument=(sum_1-bitshift(bitshift(sum_1,-shift),shift))>0 && (~sign_1);
            case 'rne'
                truncated = bitshift(sum_1, -shift);
                remainder = sum_1 - bitshift(truncated, shift);
                half = bitshift(1, shift-1);
                ulpAdjustment = (remainder > half) || (remainder == half && bitand(truncated, 1));
            otherwise
                % truncation
        end
        sum_1=bitshift(sum_1,-shift)+uint64(ulpAdjument);
    elseif max_exp_2<max_exp  
        % round down odd indexed product sum
        neab=neab+eab_2; % adjust neab
        sum_1=bitshift(sum_1,eab_2);
        sum_2=bitshift(sum_2,eab_2);
        switch align_round_mode
            case 'rd' 
                ulpAdjument=(sum_2-bitshift(bitshift(sum_2,-shift),shift))>0 && sign_2;
            case 'ru'
                ulpAdjument=(sum_2-bitshift(bitshift(sum_2,-shift),shift))>0 && (~sign_2);
            case 'rne'
                truncated = bitshift(sum_2, -shift);
                remainder = sum_2 - bitshift(truncated, shift);
                half = bitshift(1, shift-1);
                ulpAdjustment = (remainder > half) || (remainder == half && bitand(truncated, 1));
            otherwise
                % truncation
        end
        sum_2=bitshift(sum_2,-shift)+uint64(ulpAdjument);
    else
        %nothing
    end
    operands=double([sum_1,sum_2]);
    prod_sum_unnorm=dot(operands,1-2*[sign_1,sign_2]);
    prod_sum_sign=prod_sum_unnorm<0;
end


%==========================================================================
%% Function: Normalisation Helper Function
% takes in aligned significand sum as integer along with neab, max_exp,
% stkbit
%==========================================================================
 
function [dbits,dexp]=norm_helper(sum_unormalised,max_exp,neab,stkbit)
    sum_unormalised_uint64=uint64(abs(sum_unormalised));
    sum_normalised=sum_unormalised/2^(23+neab+stkbit); 
    [~,total_exp]=log2(abs(sum_normalised)); total_exp=total_exp-1;
    dexp=max_exp+total_exp;
    if total_exp>0
        temp_str=dec2bin(sum_unormalised_uint64);
        dbits=[temp_str(1),'.',temp_str(2:end)];
    else % normalised even if supposed to be denormalised
        total_exp=abs(total_exp);
        sum_unormalised_uint64=bitshift(sum_unormalised_uint64,total_exp);
        temp_str=dec2bin(sum_unormalised_uint64);
        dbits=[temp_str(1),'.',temp_str(2:end)];
    end
end

%==========================================================================
%% Function: Kulisch Accumulation
%==========================================================================
function [dbits, dexp, signOut] = Kulisch_Accumulation(prod_sig, prod_exp, c, signBits,c_zero_check)

    % Include constant term if non-zero
    if ~c_zero_check
        [~, constExp] = log2(abs(c));
        prod_exp(end+1) = constExp - 1;
        prod_sig(end+1) = c / 2^prod_exp(end);
    end

    % Align significands
    [alignedExp, significandBits] = AlignSignificandBits(prod_sig);
    alignedExp = alignedExp + prod_exp;

    % Find maximum exponent
    [maxExponent, ~] = max(alignedExp);

    % Compute required shift
    maxShift = max(maxExponent - alignedExp);
    totalBitLength = 2 + 23 + maxShift;

    numTerms = numel(prod_sig);

    % Preallocate aligned bit array
    alignedBitMatrix = repmat('0', numTerms, totalBitLength);

    baseBitLength = 2 + 23;

    % Compute exponent shifts
    shiftArray = maxExponent - alignedExp;

    for idx = 1:numTerms
        shiftVal = shiftArray(idx);

        if shiftVal > 0
            % Insert leading '0.'
            alignedBitMatrix(idx, 1:2) = '0.';

            % Fill zeros between decimal point and significant bits
            if shiftVal > 1
                alignedBitMatrix(idx, 3:shiftVal+1) = '0';
            end

            % Insert shifted significand bits
            alignedBitMatrix(idx, shiftVal+2 : shiftVal + baseBitLength) = ...
                [significandBits(idx,1), significandBits(idx,3:end)];
        else
            % No shift needed
            alignedBitMatrix(idx, 1:baseBitLength) = significandBits(idx,:);
        end
    end

    alignedBitStrings = string(alignedBitMatrix);

    % Binary accumulation
    [dbits, signOut, integerPart, decimalPoint] = ...
        sumBinaryFixedBitwise(signBits, alignedBitStrings);

    % Normalisation
    [dbits, dexp] = ...
        NormalisationPostAddition(dbits, maxExponent, decimalPoint, integerPart);
    if isempty(dexp)
        dexp=0;
    end
end

%==========================================================================
%% Function: subnormal significand
%==========================================================================
function [dbits]=subnormalsignificand(dbits,shift,truncate)
CharLen=numel(dbits);
zero_app=char(zeros(1,min(CharLen,shift+1))+'0');
zero_app(2)='.';
if shift>=CharLen
    dbits=zero_app;
else
    dbits=[zero_app,dbits(1),dbits(3:CharLen)];
end
if truncate
dbits(CharLen+1:end)=[];
end
end


%==========================================================================
%% Function for summing Binary Strings as bitwise with 2s complement:
%% Called from Within CDNA_1 function
%==========================================================================
function [acc,resultSign,intpart,decimalpoint] = sumBinaryFixedBitwise(signBits, bitStrings)
% signBits  : char array like '0101...'
% bitStrings: cell array of strings like {'1.0101','0.1110',...}
% result    : summed binary string in same format

    N = length(bitStrings);

    % Remove dot and determine length
    L = length(bitStrings{1}) - 1;   % total bits without dot
    M = L - 1;                      % fractional bits

    % Accumulator with extra guard bits to prevent overflow
    guardBits = ceil(log2(N))+2;
    totalLen = L + guardBits;

    acc = zeros(1,totalLen);   % numeric bit array

    for i = 1:N

        bits = bitStrings{i};
        bits(2) = [];                  % remove dot
        b = bits - '0';                % convert to numeric row

        % Left pad with zeros to match accumulator length
        b = [zeros(1,totalLen-L) b];

        % If negative → convert to two's complement
        if signBits(i)
            b = 1 - b;                 % invert bits

            % add 1
            carry = 1;
            for k = totalLen:-1:1
                s = b(k) + carry;
                b(k) = mod(s,2);
                carry = floor(s/2);
                if carry == 0
                    break;
                end
            end
        end

        % Add to accumulator (bit-by-bit)
        carry = 0;
        for k = totalLen:-1:1
            s = acc(k) + b(k) + carry;
            acc(k) = mod(s,2);
            carry = floor(s/2);
        end
    end

    % Detect sign of result (two's complement form)
    if acc(1) == 1
        % negative → convert back from two's complement
        acc = 1 - acc;
        carry = 1;
        for k = totalLen:-1:1
            s = acc(k) + carry;
            acc(k) = mod(s,2);
            carry = floor(s/2);
            if carry == 0
                break;
            end
        end
        resultSign = 1;
    else
        resultSign = 0;
    end

    % Remove guard bits
    %acc = acc(end-L+1:end);
    acc= char(acc + '0');
    intpart=bin2dec(acc(1:guardBits+1));
    intpartbin=dec2bin(intpart);
    acc=[intpartbin,'.',acc(guardBits+2:end)];
    decimalpoint=numel(intpartbin)+1;
    % Insert decimal point
    
end


%=====================================================
%% Accumulation of Binary Char Arrays
%====================================================
function [total] = AccBinStrs(BitCharArray, SignBits)
%SUMBINARYSTRINGS Sum binary strings with fractional parts and signs
%
% Inputs:
%   BitCharArray : n×1 string array or cell array of chars, e.g. ["1.101", "0.011"]
%   SignBits     : n×1 numeric array, 0=positive, 1=negative
%
% Output:
%   total        : signed sum of all binary numbers

% Ensure string array
x=bin2dec([BitCharArray(:,1),BitCharArray(:,3:end)]);
x=x/(2^(numel(BitCharArray(1,:))-2));
SignBits=1-2*SignBits;
total=dot(x,SignBits);


end


%% ========================================================================
%  Fraction to Bins
%  ========================================================================
function binStr = frac2bins(x, nBits)
    % binStr = '';
    % for i = 1:nBits
    %     x = x * 2;
    %     if x >= 1
    %         binStr = [binStr '1'];
    %         x = x - 1;
    %     else
    %         binStr = [binStr '0'];
    %     end
    % end


  binStr = repmat('0', 1, nBits);

    for i = 1:nBits
        x = x * 2;
        if x >= 1
            binStr(i) = '1';
            x = x - 1;
        end
    end


end




%% ========================================================================
%%  Sub-Function: IEEE Rounding (RD/RU/RNE/RZ)
% ========================================================================
function [outbits, outexp] = ieeeround(inbits, rndmode, NoManBits, signbit, inexp)
%IEEEROUND IEEE-754 Rounding Operation
%
%   [outbits, outexp] = ieeeround(inbits, rndmode, NoManBits, signbit, inexp)
%
%   Applies IEEE-compliant rounding modes to a binary mantissa string.
%
%   Inputs:
%       inbits    - Input mantissa bits (char array)
%       rndmode   - Rounding mode string:
%                   'RNE' / 'rne' : Round-to-nearest (ties to even)
%                   'RD'  / 'rd'  : Round-toward-negative
%                   'RU'  / 'ru'  : Round-toward-positive
%       NoManBits - Number of mantissa bits
%       signbit   - Sign bit (0 = positive, 1 = negative)
%       inexp     - Exponent value before rounding
%
%   Outputs:
%       outbits - Rounded mantissa bits
%       outexp  - Updated exponent after rounding (if overflow occurs)
%
%   Dependencies:
%       - computgrtbits()
%       - NormPostAddULP()
%
% -------------------------------------------------------------------------
    OutCharLen = 2 + NoManBits;
    outexp     = inexp;
    inbits2=inbits;
    % Truncate bits beyond mantissa length
    inbits(:, OutCharLen+1:end) = [];
    outbits    = inbits;
    grtbits=any(inbits2(OutCharLen+1:end) == '1');  % 1 if any bit in position 3 onward is '1'

    % Select rounding mode
    switch rndmode
        
        case {"rd"} % Round-toward-negative
            if signbit && grtbits > 0
                [outbits, outexp] = NormPostAddULP(inbits, inexp, NoManBits);
            end

        case {"ru"} % Round-toward-positive
            if ~signbit && grtbits > 0
                [outbits, outexp] = NormPostAddULP(inbits, inexp, NoManBits);
            end

        case {"rne"} % Round-to-nearest
            % compute the grtbits with sticky bit
            grtbits   = computgrtbits(NoManBits, inbits2);
            if grtbits > 4 || (grtbits==4 && outbits(end)=='1')
                [outbits, outexp] = NormPostAddULP(inbits, inexp, NoManBits);
            end

        otherwise
            warning("Unknown rounding mode string: %s", rndmode);
    end
end

%-----------------------------------------
%% 
%-----------------------------------------
function [outbits,outexp]=NormPostAddULP(instr,inexp,nomanbits)
total_unormalised=bin2dec([instr(1),instr(3:end)])+1;
total=total_unormalised/2^nomanbits;
intpart=abs(fix(total));
fracpart=abs(total)-abs(intpart);
fracstr=frac2bins(fracpart, nomanbits);
intpartstr=dec2bin(intpart);
decpointidx=numel(intpartstr)+1;
outstr = [intpartstr, '.', fracstr];


 % total_exp=floor(log2(intpart));
 % outexp2=inexp+total_exp;
 % if total_exp>0
 %        temp_str=dec2bin(total_unormalised);
 %        outbits2=[temp_str(1),'.',temp_str(2:end)];
 % else
 %        total_exp=abs(total_exp);
 %        total_unormalised_uint32=bitshift(uint32(total_unormalised),total_exp);
 %        temp_str=dec2bin(total_unormalised_uint32);
 %        outbits2=[temp_str(1),'.',temp_str(2:end)];
 % end
 % 




[outbits,outexp]=NormalisationPostAddition(outstr,inexp,decpointidx,intpart);
if outexp~=inexp

 %   disp('implement another rounding');
else
    outbits(:,2+nomanbits+1:end)=[];
end



end


%--------------------------------------------------------------------
%% compute GRBT Bits Function
%--------------------------------------------------------------------
function [guardbitsdec]=computgrtbits(NoManBits,AlignBits)
K=numel(AlignBits(:,1));
CharLength=numel(AlignBits(1,:));
OutCharLength=2+NoManBits;

if CharLength > OutCharLength
    grtBits = AlignBits(min(2,K), OutCharLength+1:end);
else
    grtBits = '0';
end

grtBits = [grtBits, repmat('0', 1, max(0, 3 - numel(grtBits)))];
guard = grtBits(1:2);
guard(3)='0';
sticky = any(grtBits(3:end) == '1');  % 1 if any bit in position 3 onward is '1'

guardbitsdec = bin2dec(guard) + sticky;

end

%---------------------------------------------
%%   Normalisation Post Addition
%---------------------------------------------
function [d_in_bits,final_exp_actual]=NormalisationPostAddition(resultStr,largest_exp,decimalpoint,intpart) 

 charCount=numel(resultStr);
 exp_shift_result=0;
 bit_before_dec_point=decimalpoint-1;
 
 
 % carry occured
if intpart==1
    % no change
    d_in_bits=resultStr;

elseif intpart>1
    % shift to the right and increase the exponent
  exp_shift_result=bit_before_dec_point-1;
  %d_in_bits=['1','.',split_array{1}(2:end),split_array{2}];
  d_in_bits=['1','.',resultStr([2:decimalpoint-1,decimalpoint+1:end])];
else
% small and therefore left shift and decrease the exponent
    %firstOne = find(split_array{2} == '1', 1, 'first');
    firstOne = find(resultStr(decimalpoint+1:end) == '1', 1, 'first');
    exp_shift_result=-firstOne;
    if isempty(firstOne)
    % all zeros no firstOne
        d_in_bits=resultStr; % no change
    else
         %d_in_bits=['1.',split_array{2}(firstOne+1:end)];
         d_in_bits=['1.',resultStr(firstOne+decimalpoint+1:end)];

         extracharappend=charCount-numel(d_in_bits);
         d_in_bits=[d_in_bits,char(zeros(1,extracharappend)+'0')];
    end 
end
final_exp_actual=largest_exp+exp_shift_result;

end


%---------------------------------------------
%% Takes in products significand and align based on exponents
%% outputs aligned significands as array of characters
%---------------------------------------------

function [exp_unbiased, BitCharArray] = AlignSignificandBits(x)

    
    N=numel(x);
    x=single(x);

    u = typecast(x, 'uint32');

    exp_raw  = bitand(bitshift(u, -23), uint32(255));     % 8 bits
    frac     = bitand(u, uint32(2^23 - 1));               % 23 bits
    bias     = 127;

    implicit_bit = uint32(2^23);  % 1 << 23

    normal_mask = exp_raw ~= 0;
    implicit = zeros(size(u),'uint32');
    implicit(normal_mask) = implicit_bit;

    full_sig = implicit + frac;   % 24 bits
    significand_bits = dec2bin(full_sig, 24);
    BitCharArray=[significand_bits(:,1),repmat('.',N,1),significand_bits(:,2:end)];
    exp_unbiased = double(exp_raw) - bias;
    
end

%---------------------------------------------
%% Takes in products significand and align based on exponents
%% outputs aligned significands as usigned integers
%---------------------------------------------
function [maxExp, alignedSig] = AlignSignficand(x, xExp, c, neab, stkbit,minexplimit,c_min_exp_limit)
%FPBITS_IEEE2 Extract and align IEEE-754 significands with exponents
%
% Inputs:
%   x       : array of values (single precision assumed)
%   xExp    : corresponding unbiased exponents for x
%   c       : optional scalar to include in alignment
%   neab    : integer counter (used for extra bit allowance)
%   stkbit  : boolean flag to include sticky bits
%
% Outputs:
%   maxExp     : maximum exponent after alignment
%   alignedSig : aligned significands (uint32)
%   neab       : updated extra bit allowance

    % Take absolute values
    x = abs(x);

    %% Constants
    FP32_BIAS      = int16(127);
    FP32_IMPLICIT = bitshift(uint32(1), 23);  % 2^23         % 1 << 23
    FP32_FRAC_MASK = FP32_IMPLICIT - 1;        % mask for lower 23 bits

    %% === Product path ===
    expVals    = int16(xExp);                  % unbiased exponents
    sigVals    = uint32(x .* 8388608);        % convert to 23-bit significands
    
    %% === Optional scalar 'c' ===
    if c ~= 0
        cUint32  = typecast(single(c), 'uint32');
        rawExpC  = bitand(bitshift(cUint32, -23), uint32(255));
        expC     = int16(rawExpC) - FP32_BIAS;
        expC(expC == -127) = -126;           % subnormal correction

        fracC    = bitand(cUint32, FP32_FRAC_MASK);
        sigC     = fracC + uint32(rawExpC ~= 0) * FP32_IMPLICIT;   
    else
       if ~c_min_exp_limit
        expC     = [];
        sigC     = [];
       else
        expC     = -126;
        sigC     = 0;

       end
    end
    %% === Combine products with optional scalar 'c' ===
    expVals    = [expVals, expC];
    sigVals    = [sigVals, sigC];
    
    

    %% === Alignment ===
    maxExp     = max(expVals);

    %% == putting a limit on the max exponent =====
    if exist('minexplimit')
        if maxExp<minexplimit % considering bf16 
            maxExp=minexplimit;
        end
    end
    shiftExps  = maxExp - expVals;
    
    if neab ~= 0
        sigVals = bitshift(sigVals, neab);
    end

    %% === Sticky bit handling ===
    validshifts_2 = shiftExps<=31; % otherwise octave miss up
            if stkbit
                bitlen=24+neab;
                validshifts= shiftExps<=bitlen; %27 for H100
                % invalid shifts lost all bits
                lostMask(validshifts)   = bitshift(uint32(1), shiftExps(validshifts)) - 1;
                lostMask(~validshifts)  = 2^(bitlen)-1;
                lostBits   = bitand(sigVals, lostMask) ~= 0;
                
                sigVals(validshifts_2)  = bitshift(sigVals(validshifts_2), -shiftExps(validshifts_2));
                sigVals(~validshifts_2) = 0;
                alignedSig = sigVals * uint32(2^stkbit) + uint32(lostBits);
            else

                alignedSig(validshifts_2) = bitshift(sigVals(validshifts_2), -shiftExps(validshifts_2));
                alignedSig(~validshifts_2) = 0; 
            end

end



