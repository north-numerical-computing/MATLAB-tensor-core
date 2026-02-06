function d = Generic_BFMA_TC(NoManBitsPrd, OutRoundMode, neab, stkbitenabled,NoManBitsOut,NoExpBitsOut,a_block,b_block,c,NoExpBitsIn)
% This version computes product via significands separately

%% Initialization
emin_output=1-(2^(NoExpBitsOut-1)-1);
emin_input=1-(2^(NoExpBitsIn-1)-1);
    
% remove zero from the array
prod_zeroIdxs=(a_block==0)|(b_block==0);
a_block(prod_zeroIdxs)=[];   
b_block(prod_zeroIdxs)=[];

% if all are zero, return with d=0; 
if isempty(a_block) && c==0
    d=0;
return
end

% input a and b exponent and significands computation
a_block_abs=abs(a_block);       b_block_abs=abs(b_block);
[~,a_exp]=log2(a_block_abs);    a_exp=a_exp-1; a_exp=max(a_exp,emin_input);
[~,b_exp]=log2(b_block_abs);    b_exp=b_exp-1; b_exp=max(b_exp,emin_input);
a_sig=pow2(a_block,-a_exp);     b_sig=pow2(b_block,-b_exp);
prod_sig=a_sig.*b_sig;          prod_exp=a_exp+b_exp; 

sign_bits = (prod_sig < 0);
if c ~= 0
    sign_bits(end+1) = (c < 0);
end       
%% -------------------------
  %  ACCUMULATION & ALIGNMENT
%  -------------------------
[max_exp_unbiased, align_sigs,neab] = fpbits_IEEE2(prod_sig,prod_exp,c,neab,stkbitenabled);
sum_unormalised=dot(double(align_sigs),(1-2*(sign_bits)));
sum_unormalised_uint64=uint64(abs(sum_unormalised));
sum_normalised=sum_unormalised/2^(NoManBitsPrd+neab+stkbitenabled);   
sOut=sum_normalised<0;
        
    if sum_unormalised==0
        d=0;
        return
    end
    
    
    
    %=====================================================================
    %% Normalisation
    %=====================================================================
    [~,total_exp]=log2(abs(sum_normalised)); total_exp=total_exp-1;
    dexp=max_exp_unbiased+total_exp;
    if total_exp>0
        temp_str=dec2bin(sum_unormalised_uint64);
        dbits=[temp_str(1),'.',temp_str(2:end)];
    else % normalised even if supposed to be denormalised
        total_exp=abs(total_exp);
        sum_unormalised_uint64=bitshift(sum_unormalised_uint64,total_exp);
        temp_str=dec2bin(sum_unormalised_uint64);
        dbits=[temp_str(1),'.',temp_str(2:end)];
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
   
    
    % Encode exponent (bias applied)
    dexp = dexp + (2^(NoExpBitsOut - 1) - 1);
    %-----------------------------------
    % checking Inf/-Inf
    %-----------------------------------
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


% subnormalsignificand

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



function [maxExp, alignedSig, neab] = fpbits_IEEE2(x, xExp, c, neab, stkbit)
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
    maxExpVal  = max(expVals);                % maximum exponent among products
    idxMaxExp  = (expVals == maxExpVal);      
    maxProduct = max(x(idxMaxExp));           % largest product for normalization

    %% === Optional scalar 'c' ===
    if c ~= 0
        cUint32  = typecast(single(c), 'uint32');

        rawExpC  = bitand(bitshift(cUint32, -23), uint32(255));
        expC     = int16(rawExpC) - FP32_BIAS;
        expC(expC == -127) = -126;           % subnormal correction

        fracC    = bitand(cUint32, FP32_FRAC_MASK);
        sigC     = fracC + uint32(rawExpC ~= 0) * FP32_IMPLICIT;
        if ~isempty(maxExpVal)
        spcFlag  = maxExpVal >= expC;
        else
          spcFlag = false;
        end% extra bit allowance
    else
        spcFlag  = ~isempty(maxExpVal);
        expC     = int16.empty;
        sigC     = uint32.empty;
    end

    % Only allow extra bit if largest product >= 2
    
     spcFlag = spcFlag && (maxProduct >= 2);
    
    %% === Normalize significands >= 2 ===
    exceedMask = (x >= 2);
    if any(exceedMask)
        sigVals(exceedMask)   = bitshift(sigVals(exceedMask), -1);
        expVals(exceedMask)   = expVals(exceedMask) + 1;
    end

    %% === Combine products with optional scalar 'c' ===
    expVals    = [expVals, expC];
    sigVals    = [sigVals, sigC];
    neab       = neab + spcFlag;

    %% === Alignment ===
    maxExp     = max(expVals);
    shiftExps  = maxExp - expVals;

    if neab ~= 0
        sigVals = bitshift(sigVals, neab);
    end

    %% === Sticky bit handling ===
    if stkbit
        lostMask   = bitshift(uint32(1), shiftExps) - 1;
        lostBits   = bitand(sigVals, lostMask) ~= 0;

        sigVals    = bitshift(sigVals, -shiftExps);
        alignedSig = sigVals * uint32(2^stkbit) + uint32(lostBits);
    else
        alignedSig = bitshift(sigVals, -shiftExps);
    end

end



