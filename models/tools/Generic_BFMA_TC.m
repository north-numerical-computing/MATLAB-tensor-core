function d = Generic_BFMA_TC(r, NoExpBitsPrd, NoManBitsPrd, OutRoundMode, neab, stkbitenabled,NoManBitsOut,NoExpBitsOut,a_block,b_block,c,special_case,NoExpBitsIn)
%BLOCKFMACALL Iterative Block FMA Operation
%
%   [d, dbits, dexp, sOut] = BlockFMACall(r, NoExpBits, NoManBits, OutRoundMode, neab, stkbitenabled)
%
%   Performs a block fused multiply-add (FMA) operation on the input vector
%   of floating-point operands. Operands are aligned, accumulated, and
%   rounded according to IEEE floating-point rules.
%
%   Inputs:
%       r             - Vector of operands (already in target precision domain)
%       NoExpBitsPrd     - Number of exponent bits in target format
%       NoManBitsPrd     - Number of mantissa bits (excluding implicit bit)
%       OutRoundMode  - Rounding mode (e.g. 'RNE', 'RTZ', 'RUP', 'RDN')
%       neab          - Number of effective alignment bits (Inf = full precision)
%       stkbitenabled - Enable sticky-bit handling for truncated bits
%
%   Outputs:
%       d     - Final accumulated result (in numeric value)

% -------------------------------------------------------------------------
% if accumulation over a single column of B has generated previously Inf/-inf
% since the intial if check before for loop over columns has been passed
    if any(isnan(r)) || any(isinf(r))
        d=sum(r);
        return
    end



    % Initialise outputs
    d     = 0;
    dbits = [];
    dexp  = 0;
    sOut  = 0; 
    K = numel(r); % Number of operands
    emin=1-(2^(NoExpBitsOut-1)-1);
    %% products recomputed for denormalised product case
    r2=a_block.*b_block; % second set of products
    a_block(r2==0)=[];   
    b_block(r2==0)=[];
    
    %%  check for denormalised product 
    if K>1 % if only one element, dont bother runing this code
    if special_case==1 
      
        [a_norm,b_norm,a_exp,b_exp]=norm_exp_log2(a_block,b_block,NoExpBitsIn);
        prod_norm = a_norm.*b_norm;
        if c~=0
            c_exp=floor(log2(abs(c)));
            %c_norm=c/2^c_exp;
        else
            c_exp=[];
            %c_norm=[];
        end
            prod_exp=a_exp+b_exp;
            
            % 1: c greater than other exponent, no special case
            if ~isempty(c_exp)
                if all(c_exp>prod_exp) 
                    special_case=0;
                    
                end
            end
            maxexp=max(prod_exp);
            idx=find(prod_exp==maxexp);
            prod_norm_2=prod_norm(idx);
            if abs(max(abs(prod_norm_2)))<2
                special_case=0;
            end
        end
        spc=special_case; 
    end
    
    %% extract IEEE 754 bits format from the product and c
    [ExpBitsArray, BitCharArray] = fpbits_IEEE(r, NoManBitsPrd);
    SignBits=double(r<0);
    %% if only one element is present in the provided block, dont compute further
    %-----------------------------------------------------------------
    if K==1
       dexp=ExpBitsArray(1);
       dbits=BitCharArray(1,:);
            if strcmp(OutRoundMode,'rz') 
                 dbits(3+NoManBitsOut:end)=[]; % truncation
            else
                [dbits,dexp] = ieeeround(dbits, OutRoundMode, NoManBitsOut, sOut, dexp);
            end
    
    if dexp<emin
        min_shift=dexp-emin;
        [dbits]=subnormalsignificand(dbits,abs(min_shift),1);
        dexp=emin;
    end

       d= ((dbits(1)-'0')+bin2dec(dbits(3:end))*2^(-NoManBitsOut))*2^dexp; 
        if SignBits(1)==1
               d=-d;
        end
    
       return
    end    
    %----------------------------------------------------------------
    
     %% ALIGNMENT OF SIGNIFICANDS (Optimized)

        maxexpshift = ExpBitsArray(1) - ExpBitsArray(end);
        totalLength = 2 + spc + NoManBitsPrd + maxexpshift;
        
        % Preallocate AlignBits as char array of '0's
        AlignBits = repmat('0', K, totalLength);
        
        % Place first operand directly
        lenBitChar = 2 + NoManBitsPrd;
        AlignBits(1, 1:lenBitChar) = BitCharArray(1,:);
        
        % Compute shifts for all subsequent operands
        expshiftArr = ExpBitsArray(1) - ExpBitsArray(2:end);  % vector of shifts
        
        for k = 2:K
            shift = expshiftArr(k-1);
            
            if shift > 0
                % Place '0.' at start
                AlignBits(k, 1:2) = '0.';
                
                % Fill remaining leading zeros if needed
                if shift > 1
                    AlignBits(k, 3:shift+1) = '0';
                end
                
                % Copy the significant bits from BitCharArray
                AlignBits(k, shift+2 : shift + lenBitChar) = [BitCharArray(k,1), BitCharArray(k,3:end)];
            else
                % No shift, copy directly
                AlignBits(k, 1:lenBitChar) = BitCharArray(k,:);
            end
        end
    
        
   
    %%  TRUNCATION (if neab < Inf)
    neab=neab+spc;
    if neab ~= Inf
        if stkbitenabled
            stkbits = any(AlignBits(:, 2 + NoManBitsPrd + neab+1:end) == '1', 2);
            stkbits = char(stkbits + '0');
            AlignBits(:, 2 + NoManBitsPrd + neab + 1:end) = [];
            AlignBits = [AlignBits,stkbits];
        else
            AlignBits(:, 2 + NoManBitsPrd + neab + 1:end) = [];
        end
    end

    %% -------------------------
    %  ACCUMULATION
    %  -------------------------
    total = AccBinStrs(AlignBits, double(SignBits));
    intpart=abs(fix(total));
    fracpart=abs(total)-abs(intpart);
    if neab==Inf
        fracstr=frac2bins(fracpart, numel(AlignBits(1,:))-2);
         
    else
        fracstr=frac2bins(fracpart, NoManBitsPrd+neab+stkbitenabled);
    end
    sOut=total<0;
    intpartstr=dec2bin(intpart);
    decimal_point=numel(intpartstr)+1;
    ResultStr = [intpartstr, '.', fracstr];
    
    
    

    %% -------------------------
    %  NORMALISATION & ROUNDING
    %  -------------------------
    [dbits, dexp] = NormalisationPostAddition(ResultStr, max(ExpBitsArray),decimal_point,intpart);
    % subnormal range, 
    if dexp<emin
        min_shift=dexp-emin;
        [dbits]=subnormalsignificand(dbits,abs(min_shift),0);
        dexp=emin;
    end   
    
    if strcmp(OutRoundMode,'rz') 
         dbits(3+NoManBitsOut:end)=[]; % truncation
    else
        [dbits,dexp] = ieeeround(dbits, OutRoundMode, NoManBitsOut, sOut, dexp);
    end
    
    if dexp<emin
        min_shift=dexp-emin;
        [dbits]=subnormalsignificand(dbits,abs(min_shift),1);
        dexp=emin;
    end 
    if isempty(dexp)
    dexp=0;
    end
    % compute decimal 
    d= ((dbits(1)-'0')+bin2dec(dbits(3:end))*2^(-NoManBitsOut))*2^dexp;
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
         % uncomment to see dexp_bits 
        % dexp = dec2bin(dexp, NoExpBitsOut);
        dbits=[]; % empty on purpose, 
    return
    end

% see the exponent bits in IEEE 754 format
    
    if isempty(d)
        d=0;
    end


end

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
%
% =========================================================================
function [a_norm,b_norm,a_exp,b_exp]=norm_exp_log2(a_block,b_block,NoExpBitIn)
%------------------------------------
    emin=-2^(NoExpBitIn-1)+2;
    % Compute exponents
    a_exp = floor(log2(abs(a_block)));
    a_exp(a_exp<emin)=emin;

    b_exp = floor(log2(abs(b_block)));
    b_exp(b_exp<emin)=emin;

    % Normalize blocks
    a_norm = a_block ./ 2.^a_exp;
    b_norm = b_block ./ 2.^b_exp;

%----------------------------------


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
total=bin2dec([instr(1),instr(3:end)])/2^(nomanbits)+2^(-nomanbits);
intpart=abs(fix(total));
fracpart=abs(total)-abs(intpart);
fracstr=frac2bins(fracpart, nomanbits);
intpartstr=dec2bin(intpart);
decpointidx=numel(intpartstr)+1;
outstr = [intpartstr, '.', fracstr];


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
function [exp_unbiased, BitCharArray] = fpbits_IEEE(x, NoManBits)
%IEEE754_PARTS Extract IEEE-754 exponent + significand for half or single precision.
%
%   [exp_unbiased, significand_bits] = ieee754_parts(x, 'single')
%   [exp_unbiased, significand_bits] = ieee754_parts(x, 'half')
%
%   exp_unbiased     : unbiased exponent (decimal)
%   significand_bits : bit strings with implicit bit included (normal)
%                      or 0.xxx… (subnormal)
%
%   Handles normals, subnormals, zeros, and preserves vectorization.

    if nargin < 2
        error('Precision must be ''single'' or ''half''.');
    end
    N=numel(x);
    if NoManBits==23
            
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
            exp_unbiased(exp_unbiased==-127)=-126;
    elseif NoManBits==10
            % IEEE754 half precision: 1 | 5 | 10
            % MATLAB stores half as uint16 internally? 
            % We use typecast on uint16 representation.
            x=half(x);
            
            u = typecast(x, 'uint16');

            exp_raw  = bitand(bitshift(u, -10), uint16(31));     % 5 bits
            frac     = bitand(u, uint16(2^10 - 1));              % 10 bits
            bias     = 15;

            implicit_bit = uint16(2^10);  % 1 << 10

            normal_mask = exp_raw ~= 0;
            implicit = zeros(size(u),'uint16');
            implicit(normal_mask) = implicit_bit;

            full_sig = implicit + frac;   % 11 bits
            significand_bits = dec2bin(full_sig, 11);
            BitCharArray=[significand_bits(:,1),repmat('.',N,1),significand_bits(:,2:end)];
            exp_unbiased = double(exp_raw) - bias;
            exp_unbiased(exp_unbiased==-15)=-14;

    else
            error('Precision must be ''single'' or ''half''.');
    end
end


