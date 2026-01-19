function d = Generic_BFMA_TC(NoExpBitsPrd, NoManBitsPrd, OutRoundMode, neab, stkbitenabled,NoManBitsOut,NoExpBitsOut,a_block,b_block,c,NoExpBitsIn)

       
    %% products recomputed for denormalised product case
    r2=a_block.*b_block; % second set of products
    a_block(r2==0)=[];   
    b_block(r2==0)=[];
    r2(r2==0)=[];
    if abs(c)>abs(r2)
        special_case=0;
    else
        special_case=1;
    end
    % if accumulation over a single column of B has generated previously Inf/-inf
    % since the intial if check before for loop over columns has been passed
    if any(isnan([r2,c])) || any(isinf([r2,c]))
        d=sum(r);
        return
    end
    
    % Initialise outputs
    d     = 0;
    emin_output=1-(2^(NoExpBitsOut-1)-1);
    emin_input=1-(2^(NoExpBitsIn-1)-1);
    emin_product=1-(2^(NoExpBitsPrd-1)-1);
    
    %%  check for denormalised product --9:55 8/12/2025
    spc=0;
    if special_case==1 && ~isempty(r2)
                    [~,a_exp]=log2(abs(a_block)); a_exp=a_exp-1;
                    [~,b_exp]=log2(abs(b_block)); b_exp=b_exp-1;
                    a_exp_u=max(a_exp,emin_input);
                    b_exp_u=max(b_exp,emin_input);
                    
                    prod_exp=a_exp_u+b_exp_u;
                    prod_exp_ind= prod_exp==max(prod_exp);
                    prod_exp=prod_exp(prod_exp_ind);
                    
                    prod_sig=abs(r2(prod_exp_ind))./(2^prod_exp(1));
                    prod_sig=max(prod_sig);

                    [~,c_exp]=log2(abs(c));
                    c_exp=c_exp-1;
                    
                if prod_exp(1)>=(max(c_exp,emin_product)) && (prod_sig>=2) 
                    spc=1;
                end
 else
        [~,a_exp]=log2(abs(a_block)); a_exp=a_exp-1; 
        [~,b_exp]=log2(abs(b_block)); b_exp=b_exp-1;
        
end

a_sig=pow2(a_block,-a_exp);b_sig=pow2(b_block,-b_exp); 

prod_exp=a_exp+b_exp; prod_sig=a_sig.*b_sig;

if c~=0
    sign_bits=double([prod_sig,c]<0);   
else
    sign_bits=double(prod_sig<0);
end        
%% -------------------------
  %  ACCUMULATION & ALIGNMENT
    %  -------------------------
    
    neab=neab+spc;
    [max_exp_unbiased, align_sigs] = fpbits_IEEE2(prod_sig,prod_exp,c,neab,stkbitenabled);
    sum_unormalised=dot(double(align_sigs),(1-2*(sign_bits)));
    sum_unormalised_uint64=uint64(abs(sum_unormalised));
    sum_normalised=sum_unormalised/2^(NoManBitsPrd+neab+stkbitenabled);   
    sOut=sum_normalised<0;
    
    
    if sum_unormalised==0
        d=0;
        return
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 
    %% Normalisation
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~,total_exp]=log2(abs(sum_normalised)); total_exp=total_exp-1;
    dexp=max_exp_unbiased+total_exp;
    if total_exp>0
        temp_str=dec2bin(sum_unormalised_uint64);
        dbits=[temp_str(1),'.',temp_str(2:end)];
    else
        total_exp=abs(total_exp);
        sum_unormalised_uint64=bitshift(sum_unormalised_uint64,total_exp);
        temp_str=dec2bin(sum_unormalised_uint64);
        dbits=[temp_str(1),'.',temp_str(2:end)];
    end
 
    
    % subnormal range, 
    if dexp<emin_output
        min_shift=dexp-emin_output;
        [dbits]=subnormalsignificand(dbits,abs(min_shift),0);
        dexp=emin_output;
    end   
    
    if ~strcmp(OutRoundMode,'rz') 
        [dbits,dexp] = ieeeround(dbits, OutRoundMode, NoManBitsOut, sOut, double(dexp));
    end
    
    if dexp<emin_output
        min_shift=dexp-emin_output;
        [dbits]=subnormalsignificand(dbits(1:2+NoManBitsOut),abs(min_shift),1);
        dexp=emin_output;
    end 
    if isempty(dexp)
    dexp=0;
    end
    % compute decimal 
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
         % uncomment to see dexp_bits 
        % dexp = dec2bin(dexp, NoExpBitsOut);
        %dbits=[]; % empty on purpose, 
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
            BitCharArray=[significand_bits(:,1),'.'+zeros(N,1),significand_bits(:,2:end)];
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
            BitCharArray=[significand_bits(:,1),'.' + zeros(N,1),significand_bits(:,2:end)];
            exp_unbiased = double(exp_raw) - bias;
            exp_unbiased(exp_unbiased==-15)=-14;

    else
            error('Precision must be ''single'' or ''half''.');
    end
end


function [max_exp_unbiased,full_sig] = fpbits_IEEE2(x,x_exp,c,neab,stkbit)
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
            
            

            bias     = 127;
            implicit_bit = uint32(8388608);  % 1 << 23
            
            
            %% for products
            x=single(x);
            u = typecast(x, 'uint32');
            exp_raw  = bitand(bitshift(u, -23), uint32(255));     % 8 bits
            exp_unbiased = int16(exp_raw) - 127 + int16(x_exp);   % stay integer
            frac     = bitand(u,implicit_bit-1); % equal 2^23-1               % 23 bits
            full_sig = frac + uint32(exp_raw ~= 0) * implicit_bit;   
            %% c is done indepdent
            if c~=0
            xc=single(c);
            uc = typecast(xc, 'uint32');
            exp_raw_c  = bitand(bitshift(uc, -23), uint32(255));     % 8 bit
            exp_c  = int16(exp_raw_c)-127;     % 8 bits
            exp_c(exp_c == -127)=-126;
            frac_c     = bitand(uc,implicit_bit-1); % equal 2^23-1               % 23 bits
            full_sig_c = frac_c + uint32(exp_raw_c ~= 0) * implicit_bit;
            full_sig=[full_sig,full_sig_c];
            % append c_exp
            exp_unbiased=[exp_unbiased,exp_c];
            end
            
            max_exp_unbiased=max(exp_unbiased);
            % shift of the exponents
            exp_shifts = max_exp_unbiased - exp_unbiased;
            lost_mask=uint32(2.^exp_shifts-1);
            
            full_sig=bitshift(full_sig,neab);
            if stkbit
             lost_bits=bitand(full_sig,lost_mask)~=0;
             full_sig=bitshift(full_sig,-exp_shifts)*2^stkbit+uint32(lost_bits);
            else
             full_sig=bitshift(full_sig,-exp_shifts);
            end
            
           
            
end
