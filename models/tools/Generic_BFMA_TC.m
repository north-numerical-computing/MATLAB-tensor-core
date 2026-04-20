function d = Generic_BFMA_TC(NoManBitsIn, OutRoundMode, neab, stkbitenabled,NoManBitsOut,NoExpBitsOut,a_block,b_block,c,NoExpBitsIn,Model)
% This version computes product via significands separately
if ~exist("Model","var")
    Model='NVIDIA'; % default
end

%% Initialization
emin_output=1-(2^(NoExpBitsOut-1)-1);
emin_input=1-(2^(NoExpBitsIn-1)-1);
    
% Identify positions where either operand is zero
prod_zeroIdxs = (a_block == 0) | (b_block == 0);

% Check if we are using CDNA3 with FP8 formats (e5m2 or e4m3)
cdna3_fp8_check = (strcmp(Model,'CDNA3') && (NoManBitsIn == 3 || NoManBitsIn == 2));

% For all cases except CDNA3 FP8:
% remove zero-valued products to avoid unnecessary computations
if ~cdna3_fp8_check
    a_block(prod_zeroIdxs) = [];   
    b_block(prod_zeroIdxs) = [];
end

% If all products are zero and accumulation term is zero,
% return zero immediately (early exit)
c_zero_check= (c==0); % to be used later
if isempty(a_block) && c_zero_check
    d = 0;
    return
end

%==========================================================================
% input a and b exponent and significands computation
%==========================================================================
a_block_abs=abs(a_block);       b_block_abs=abs(b_block);
[~,a_exp]=log2(a_block_abs);    a_exp=a_exp-1; a_exp=max(a_exp,emin_input);
[~,b_exp]=log2(b_block_abs);    b_exp=b_exp-1; b_exp=max(b_exp,emin_input);
a_sig=pow2(a_block,-a_exp);     b_sig=pow2(b_block,-b_exp);
prod_sig=a_sig.*b_sig;          prod_exp=a_exp+b_exp; 

if cdna3_fp8_check
    prod_exp(prod_zeroIdxs)=-1024; % instead of -Inf for 0 product to avoid bitshift function in error
end

sign_bits = (prod_sig < 0);




%==========================================================================
%  Model based accumulation
%==========================================================================

%==========================================================================
%% Model 1: CDNA 1, kept everything as character array of bits
%==========================================================================
if strcmp(Model,'CDNA1')
   if ~c_zero_check
    sign_bits(end+1) = (c < 0);
   end
   [dbits,dexp,sOut]=CDNA_1(prod_sig,prod_exp,c,sign_bits,c_zero_check);

%=========================================================================================
%% Model 2: NVIDIA 
%=========================================================================================
elseif strcmp(Model,'NVIDIA')
    if ~c_zero_check
    sign_bits(end+1) = (c < 0);
    end
    [dbits, dexp, sOut] = NVIDIA(prod_sig, prod_exp, c, sign_bits,neab,stkbitenabled);

%=========================================================================================
%% Model 3: CDNA 3
%=========================================================================================

elseif strcmp(Model,'CDNA3')
    
    [dbits, dexp, sOut] = CDNA_3(prod_sig, prod_exp, c, sign_bits,cdna3_fp8_check,c_zero_check);


else
% more models come here
end    

%============================================================
% make subnormal if exponent less than minimum output exponent
%=============================================================
if dexp<emin_output
    min_shift=dexp-emin_output;
    [dbits]=subnormalsignificand(dbits,abs(min_shift),0);
    dexp=emin_output;
end   
   
%=============================================================
% Rounding 
%=============================================================
if ~strcmp(OutRoundMode,'rz') 
    [dbits,dexp] = ieeeround(dbits, OutRoundMode, NoManBitsOut, sOut, double(dexp));
end

%============================================================
% make subnormal post rounding if exponent less than minimum output exponent
%=============================================================

if dexp<emin_output
    min_shift=dexp-emin_output;
    [dbits]=subnormalsignificand(dbits(1:2+NoManBitsOut),abs(min_shift),1);
    dexp=emin_output;
end 

if isempty(dexp)
 dexp=0;
end

%======================================================================
% compute the decimal value
%======================================================================
d= ((dbits(1)-'0')+bin2dec(dbits(3:NoManBitsOut+2))*2^(-NoManBitsOut))*2^double(dexp);
if sOut==1
        d=-d;
end

%==========================================================================
% Encode exponent (bias applied)
%==========================================================================
dexp = dexp + (2^(NoExpBitsOut - 1) - 1);

%==========================================================================
%  checking Inf/-Inf
%==========================================================================
if dexp==((2^NoExpBitsOut)-1)
    d=Inf;
     if sOut==1
        d=-d;
     end

return
end

% see the exponent bits in IEEE 754 format

if isempty(d)
    d=0;
end


end
% ================ Main Function End  Here =========================

%#######################################################################
% Functions from Here are Written Below
%#######################################################################

%==========================================================================
%% Function: CDNA_3 (Short descriptive name)
% Purpose : Calls CDNA_3 arch function to emulate its feature in matlab for
% all supported input formats except fp32/fp64 where its sequential FMA
% Inputs  : 
%   - prod_sig : product_significands
%   - prod_exp : product exponents, sum of exponents of two operands
%   - c : accumulation from previous  
%   - sign_bits : sign bits for product significands
%   - fp8_check : fp8 input format in CDNA_3 is detected
%   - c_zero_check: a check if c is zero or not, already checked in the
%   main function

% Outputs :
%   - dbits : output in bits in fixed points i.e. 1.01011
%   - dexp : output exponent in powers of 2
%   -signOut: output sign bit
%==========================================================================
function [dbits, dexp, signOut] = CDNA_3(prod_sig, prod_exp, c, sign_bits,fp8_check,c_zero_check);
% declare some constants
neab=1; % single extra bit

if fp8_check
  [dbits,dexp,signOut]=CDNA3_Low(prod_sig,prod_exp,sign_bits,c,neab,c_zero_check);
else
    [dbits,dexp,signOut]=CDNA3_High(prod_sig,prod_exp,sign_bits,c,neab,c_zero_check);

end


end

%==========================================================================
%% Function: CDNA_3_High (Short descriptive name)
% Purpose : Computs the output for fp16/bf16/tf32 input format inputs. This
% is called from writhin CDNA_3 function when input format is amongst as above
% Inputs  : 
%   neab: extra alignment bits

%==========================================================================

function [dbits,dexp,signOut]=CDNA3_High(prod_sig,prod_exp,sign_bits,c,neab,c_zero_check)

[max_exp, align_sigs] = AlignSignficand(prod_sig,prod_exp,0,1,0,'CNDA3');
prod_sum_unnorm=dot(double(align_sigs),(1-2*(sign_bits)));
prod_sum_sign=prod_sum_unnorm<0;

if c_zero_check
 [dbits,dexp]=norm_helper(prod_sum_unnorm,max_exp,neab,0);
 signOut=prod_sum_sign;
 return;
else
 [~,c_exp]=log2(abs(c)); c_exp=c_exp-1; c_exp=max([c_exp,-126]); c_sig=c/2^c_exp;
 c_sig_uint=uint32(abs(c_sig)*16777216); % considering 2^(23+neab=1)=2^24
end

shift=abs(max_exp-int16(c_exp));
sign_bits=[prod_sum_sign,c<0];

round_down=0; % default false
if c_exp<max_exp
    round_down=(c_sig_uint-bitshift(bitshift(c_sig_uint,-shift),shift))>0 && c<0;
    c_sig_uint=bitshift(c_sig_uint,-shift)+uint32(round_down); % round down applied
    prod_sum_unnorm_u64=abs(prod_sum_unnorm);
else
    % increase the neab
    neab=neab+(31-24);
    prod_sum_unnorm_u64=uint64(abs(prod_sum_unnorm));
    prod_sum_unnorm_u64=bitshift(prod_sum_unnorm_u64,31-24);
    c_sig_uint=bitshift(c_sig_uint,31-24);
    % check round down parameter
    round_down=(prod_sum_unnorm_u64-bitshift(bitshift(prod_sum_unnorm_u64,-shift),shift))>0 && prod_sum_sign;
    prod_sum_unnorm_u64=bitshift(prod_sum_unnorm_u64,-shift)+uint64(round_down); % round down applied
    max_exp=c_exp;
end
   operands=double([prod_sum_unnorm_u64,c_sig_uint]);
   total_sum=dot(operands,1-2*sign_bits);
   signOut=total_sum<0;
   [dbits,dexp]=norm_helper(total_sum,max_exp,neab,0);

end

%==========================================================================
%% Function: CDNA_3_Low (Short descriptive name)
% Purpose : Computs the output for fp8 input format inputs. This
% is called from writhin CDNA_3 function when input format is amongst as above
% Inputs  : See discription in CDNA_3 and CDNA3_High for inputs discreption
%==========================================================================

function [dbits,dexp,signOut]=CDNA3_Low(prod_sig,prod_exp,sign_bits,c,neab,c_zero_check)

  % remove zero from even and odd indexed
    K=numel(prod_sig);
    odd_indices=2:2:K; even_indices=1:2:K;
    even_prod_sig  = prod_sig(even_indices);     odd_prod_sig  = prod_sig(odd_indices);
    even_sign_bits = sign_bits(even_indices);   odd_sign_bits  = sign_bits(odd_indices);
    even_prod_exp  = prod_exp(even_indices);    odd_prod_exp   = prod_exp(odd_indices);

    %% Section 1: Add even and Odd indexed product significands separately
    % even indexed products
    [even_max_exp, even_align_sigs] = AlignSignficand(even_prod_sig,even_prod_exp,0,1,0,'CNDA3');
    even_prod_sum_unnorm=dot(double(even_align_sigs),(1-2*(even_sign_bits)));
    even_prod_sum_sign=even_prod_sum_unnorm<0;
    even_prod_sum_uint32=uint32(abs(even_prod_sum_unnorm));
   
    % add odd indexed product
    [odd_max_exp, odd_align_sigs] = AlignSignficand(odd_prod_sig,odd_prod_exp,0,1,0,'CNDA3');
    odd_prod_sum_unnorm=dot(double(odd_align_sigs),(1-2*(odd_sign_bits)));
    odd_prod_sum_sign=odd_prod_sum_unnorm<0;
    odd_prod_sum_uint32=uint32(abs(odd_prod_sum_unnorm));
   
    %% Section 2: Add two product sums
    max_exp=max([even_max_exp,odd_max_exp]);
    shift=abs(even_max_exp-odd_max_exp);
    % shift>512 means one of the product sum is zero, and therefore, shift
    % must be zero
    if shift>512
        shift=0;
    end
    % unlike addition of c in this function and in CDNA3_High, we have an
    % extra elseif because zero condition is not separately checked
    if even_max_exp<max_exp
        % round down even indexed product sum
        round_down=(even_prod_sum_uint32-bitshift(bitshift(even_prod_sum_uint32,-shift),shift))>0 && even_prod_sum_sign;
        even_prod_sum_uint32=bitshift(even_prod_sum_uint32,-shift)+uint32(round_down);
    elseif odd_max_exp<max_exp  
        % round down odd indexed product sum     
        round_down=(odd_prod_sum_uint32-bitshift(bitshift(odd_prod_sum_uint32,-shift),shift))>0 && odd_prod_sum_sign;
        odd_prod_sum_uint32=bitshift(odd_prod_sum_uint32,-shift)+uint32(round_down);
    else
        %nothing
    end

    operands=double([even_prod_sum_uint32,odd_prod_sum_uint32]);
    prod_sum_unnorm=dot(operands,1-2*[even_prod_sum_sign,odd_prod_sum_sign]);
    prod_sum_sign=prod_sum_unnorm<0;
    
%% Section 3: Addition of c to product sums

if c_zero_check
 [dbits,dexp]=norm_helper(prod_sum_unnorm,max_exp,neab,0);
 signOut=prod_sum_sign;
 return;
else
 [~,c_exp]=log2(abs(c)); c_exp=c_exp-1; c_exp=max([c_exp,-126]); c_sig=c/2^c_exp;
 c_sig_uint=uint32(abs(c_sig)*16777216); % considering 2^(23+neab=1)=2^24
end

shift=abs(max_exp-int16(c_exp));
sign_bits=[prod_sum_sign,c<0];

round_down=0; % default false
if c_exp<=max_exp
    round_down=(c_sig_uint-bitshift(bitshift(c_sig_uint,-shift),shift))>0 && c<0;
    c_sig_uint=bitshift(c_sig_uint,-shift)+uint32(round_down); % round down applied
    prod_sum_unnorm_u64=abs(prod_sum_unnorm);
else
    % increase the neab
    neab=neab+(31-24);
    prod_sum_unnorm_u64=uint64(abs(prod_sum_unnorm));
    prod_sum_unnorm_u64=bitshift(prod_sum_unnorm_u64,31-24);
    c_sig_uint=bitshift(c_sig_uint,31-24);
    % check round down parameter
    round_down=(prod_sum_unnorm_u64-bitshift(bitshift(prod_sum_unnorm_u64,-shift),shift))>0 && prod_sum_sign;
    prod_sum_unnorm_u64=bitshift(prod_sum_unnorm_u64,-shift)+uint64(round_down); % round down applied
    max_exp=c_exp;
end
   operands=double([prod_sum_unnorm_u64,c_sig_uint]);
   total_sum=dot(operands,1-2*sign_bits);
   signOut=total_sum<0;
   [dbits,dexp]=norm_helper(total_sum,max_exp,neab,0);

end


%==========================================================================
%% Function: NVIDIA (Short descriptive name)
% Purpose : Simulat the NVIDIA TC behaviour for a full FMA,
% Inputs
% - stkbitenabled : for stkbit in case of two operands to this function
% like only two products with c=0 or a single product with c non-zero,
% however, not used but was for inter_leaved pattern param for fp8 TC
% access via HMMA (fp16) TC where inter-leaved pattern is to be implemented
% and c is added at the end with RNE
%==========================================================================
function [dbits, dexp, signOut] = NVIDIA(prod_sig, prod_exp, c, sign_bits,neab,stkbitenabled)

    [max_exp, align_sigs] = AlignSignficand(prod_sig,prod_exp,c,neab,stkbitenabled,'NVIDIA');
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
%% Function: Normalisation Helper Function
% Description : takes in unnormalised integer sum and normalises it and
% outputs the sum as fixed point in form of bits with a decimal character
% and exponent in powers of 2
% for example: dbits=1.00110110 x 2^dexp

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
%% Function: CDNA_1
% Purpose : Simulat the CDNA_1 Matrix Core behaviour for a full FMA,
% Perform correctly rounded sum, we are not sure if its Kulisch
% accumulation or correctly rounded accumulation as it output identical
% output, 
% So the implementation is such that it keeps all bits and then round the
% final sum to fp32 via round to nearest ties to even

%==========================================================================
function [dbits, dexp, signOut] = CDNA_1(prod_sig, prod_exp, c, signBits,c_zero_check)

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

end


%==========================================================================
% Function: subnormal significand
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
% Function for summing Binary Strings as bitwise with 2s complement:
% Called from Within CDNA_1 function
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
%  Sub-Function: ieeeround
% ========================================================================
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
%
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
%------------- compute GRBT Bits Function ---------------------------
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
%
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



%--------------------------------------------------------------------
%%
%--------------------------------------------------------------------
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



function [maxExp, alignedSig] = AlignSignficand(x, xExp, c, neab, stkbit,Model)
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
       
        expC     = [];
        sigC     = [];
    end
    %% === Combine products with optional scalar 'c' ===
    expVals    = [expVals, expC];
    sigVals    = [sigVals, sigC];
    
    

    %% === Alignment ===
    maxExp     = max(expVals);
    
    %% == putting a limit on the max exponent =====
    if strcmp(Model,'NVIDIA')
        if maxExp<-133 % considering bf16 
            maxExp=-133;
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



