function D=GEMM(alpha, A, B, beta, C, informat, outformat, params)


    %% Check matrix multiplication compatibility
    [M, K1] = size(A);
    [K2, N] = size(B);
    
    if K1 ~= K2
        error('Matrix dimensions are not compatible for multiplication: A is %dx%d, B is %dx%d.', M, K1, K2, N);
    end
    if isempty(C) | ~exist('C','var')
        C=zeros(M,N);
    elseif C==0
        C=zeros(M,N);
    else
        % nothing
    end

    
    % CPFloat settings for the default input format.
    def_inopts.format = 'fp16';
    def_inopts.round  = 1; 
    def_inopts.infinity=1;

    % CPFloat settings for the default output format.
    def_outopts.format = 'fp32';
    def_outopts.round  = 4;
    def_outopts.infinity=1;
    % Set up the model according to the formats specified.
    if ismember(informat, {'fp16','half','binary16'})
        def_inopts.round=1;
        if exist('outformat', 'var')
            def_outopts.format = outformat;
            if ismember(outformat, {'fp16','binary16','half'})
                def_outopts.round=1;
            end  
        end
    elseif ismember(informat, {'tf32', 'tensorfloat32'})
        def_inopts.format=informat;
        def_inopts.round = 4;
        def_outopts.format='fp32';
    elseif ismember(informat, {'fp8-e5m2','fp8-e4m3','e5m2','e4m3'}) 
        if ismember(informat,{'e5m2','e4m3'})
            informat=upper(informat);
        end
        def_inopts.format=informat;
        def_inopts.round = 1;
    
        if exist('outformat', 'var')
            def_outopts.format = outformat;
            if ismember(outformat, {'fp16','binary16','half'})
                def_outopts.round=1;
            end
        end
    else % bfloat16
        def_inopts.format=informat;
        def_inopts.round = 1;
    end
    
    % Inputs are rounded to their respective precision via CPFloat
    if exist('cpfloat', 'file')
        A=cpfloat(alpha*A, def_inopts);
        B=cpfloat(B, def_inopts);
        C=cpfloat(beta*C, def_outopts);
    else
        fprintf(['CPFloat lib. is not used. Input matrices A, B' ...
            ' and C are assumed to be rounded to their respective' ...
            ' precisions.\n']);
    end

    % ---------------------------------------------------------
    % Initialize D
    % ---------------------------------------------------------
    D = zeros(M, N);

    %% --------------------------------------------------------
    % Extracting features and initializing constants
    % ---------------------------------------------------------
    fmt = fpformatinfo(outformat);
    NoManBitsOut = fmt.manBits;
    NoExpBitsOut = fmt.expBits;
    fmt = fpformatinfo(informat);
    NoExpBitsIn = fmt.expBits;
    NoManBitsPrd = 23; % assuming single precision product,
    NoExpBitsPrd = 8;  % assuming single precision product
    %% --------------------------------------------------------
    % Model parameter extraction
    % ---------------------------------------------------------
    nfma           = params.fma;
    neab           = params.neab;
    OutRoundMode   = lower(params.frmode);
        inter_pattern = 0;        % default
    if isfield(params,'inter_pattern'), inter_pattern = params.inter_pattern; end

        stkbitenabled = 0;        % default
    if isfield(params,'stkbitenabled'), stkbitenabled = params.stkbitenabled; end

    %% --------------------------------------------------------
    % Special cases
    % ---------------------------------------------------------

    % Special case 1: interleaved pattern → C added with different rounding mode
        % Fixed to RN according to H200/H100/B200
        cOutRoundMode = 'rne';
        % Check interleaved pattern
        if inter_pattern==1 
            nfma=2*nfma;
        end
        seq  = 0:(nfma-1);
        seq1 = seq(mod(seq, 4) < 2);
        seq1 = seq1 + 1;
        seq  = seq + 1;
        seq2 = seq(seq1 + 2);
    
    % Special case 2: Ada / L40S where number of alignment bits is negative for fp8
    if neab < 0
        if NoManBitsOut == NoManBitsPrd
            % Special case of Ada/L40S
            NoManBitsOut = NoManBitsOut + neab;
        end
    end

    %% --------------------------------------------------------
    % Flatten/reshape A to row vector and compute K
    % ---------------------------------------------------------
    a = reshape(A(1, :), 1, []);
    K = numel(a);              % Number of elements
    remainder = mod(K, nfma);  % Distance from next multiple of nfma
    if remainder~=0
        pad_size = nfma - remainder;
    else
        pad_size=0;
    end
    


if M>2 && exist('ver','file') && ~isempty(ver('parallel')) && exist('feature','file') && feature('numcores') > 1
%--------------------------------------------------------------------
%%      Parallel Computing Toolbox
%--------------------------------------------------------------------

                % Toolbox exists and multiple cores available
                    %disp('Running parallel version...');
                
                % Optionally, make sure a parallel pool is open
                    if isempty(gcp('nocreate'))
                        parpool;  % opens default number of workers
                    end
        
                    %% Parallel One Recursive Addition of TC Results 
                    parfor m = 1:M                             % rows of A
                        
                        a = [reshape(A(m, :), 1, []),zeros(1, pad_size)];        % take m-th row of A
            
                        
                        
                        d=0;
                        for n = 1:N                         % columns of B
                
                        % -------------------------------------------------------------
                        % Pad to make length an integral multiple of nfma
                        % -------------------------------------------------------------
                        b = [reshape(B(:, n), 1, []),zeros(1,pad_size)];    % take n-th column of B
                        c = C(m, n);                    % element C(m, n)
                        
                        
                        
                        K = numel(a);
                        r = a .* b;                     % elementwise multiply
                        
                        combined=[r,c];
                        if any(isnan(combined)) || any(isinf(combined))
                            if any(isnan(combined))
                                D(m,n) = NaN;           % NaN takes priority
                            elseif any(combined == Inf) && any(combined == -Inf)
                                D(m,n) = NaN;           % mixed +Inf and -Inf → ambiguous
                            elseif any(combined == Inf)
                                D(m,n) = Inf;           % only +Inf
                            elseif any(combined == -Inf)
                                D(m,n) = -Inf;          % only -Inf
                            end

                
                        % -------------------------------------------------------------
                        % Recursive accumulation via TC blocks
                        % -------------------------------------------------------------
                        else % if not nan, or inf
                
                            for k = 1 : K / nfma
                                
                                % ---------------------------------------------------------
                                % Extract block of nfma elements
                                % ---------------------------------------------------------
                                in_block = r((k - 1) * nfma + 1 : k * nfma);
                                a_block =  a((k - 1) * nfma + 1 : k * nfma);
                                b_block =  b((k - 1) * nfma + 1 : k * nfma);
                                
                                % ---------------------------------------------------------
                                % Special case: interleaved pattern (H100 / B200 / H200)
                                % ---------------------------------------------------------
                                if inter_pattern
                                    
                                    %% ------------------ First interleaved block ------------------
                                    in_block_1 = in_block(seq1);
                                    in_block_2 = [in_block(seq2), 0];
                                    
                                    a_block_1 = a_block(seq1);
                                    b_block_1 = b_block(seq1);
                                    
                                    % Sort by magnitude
                                    [~, sort_ord] = sort(abs(in_block_1), 'descend');
                                    in_block_1 = in_block_1(sort_ord);
                                    
                                    % Special case: because c=0 here
                                    special_case = 1;
                                    
                                    % Remove zeros
                                    in_block_1(in_block_1 == 0) = [];
                                    
                                    % First half cycle
                                    if ~isempty(in_block_1)
                                        d1 = Generic_BFMA_TC( ...
                                            in_block_1, NoExpBitsPrd, NoManBitsPrd, ...
                                            OutRoundMode, neab, stkbitenabled, ...
                                            NoManBitsOut, NoExpBitsOut, ...
                                            a_block_1, b_block_1, 0, special_case,NoExpBitsIn);
                                    else
                                        d1 = 0;
                                    end
                                    
                                    %% ------------------ Second interleaved block ------------------
                                    
                                    % Re-check special-case flag
                                    if abs(d1) > abs(in_block_2)
                                        special_case = 0;
                                    else
                                        special_case = 1;
                                    end
                                    
                                    % Insert previous output into block
                                    in_block_2(end) = d1;
                                    
                                    % Sort again
                                    [~, sort_ord] = sort(abs(in_block_2), 'descend');
                                    in_block_2 = in_block_2(sort_ord);
                                    
                                    a_block_2 = a_block(seq2);
                                    b_block_2 = b_block(seq2);
                                    
                                    % Remove zeros
                                    in_block_2(in_block_2 == 0) = [];
                                    
                                    % Second half cycle
                                    if ~isempty(in_block_2)
                                        d2 = Generic_BFMA_TC( ...
                                            in_block_2, NoExpBitsPrd, NoManBitsPrd, ...
                                            OutRoundMode, neab, stkbitenabled, ...
                                            NoManBitsOut, NoExpBitsOut, ...
                                            a_block_2, b_block_2, d1, special_case,NoExpBitsIn);
                                    else
                                        d2 = 0;
                                    end
                                    
                                    %% ------------------ Final addition with C ------------------
                                    
                                    in_block_3 = [d2, c];
                                    
                                    [~, sort_ord] = sort(abs(in_block_3), 'descend');
                                    in_block_3 = in_block_3(sort_ord);
                                    
                                    in_block_3(in_block_3 == 0) = [];
                                    % neab=2, stikybitenabled=1, prd=output precsision
                                    if ~isempty(in_block_3)
                                        d = Generic_BFMA_TC( ...
                                            in_block_3, NoExpBitsOut, NoManBitsOut, ...
                                            cOutRoundMode, 2, 1, ...
                                            NoManBitsOut, NoExpBitsOut, ...
                                            [], [], 0, 0);
                                        
                                    else
                                        d = 0;
                                    end
                                    
                                    c = d;     % update accumulator
                                
                            % ---------------------------------------------------------
                            % Non-interleaved pattern (standard TC path)
                            % ---------------------------------------------------------
                            else
                                
                                % Detect special-case
                                if abs(c) > abs(in_block)
                                    special_case = 0;
                                else
                                    special_case = 1;
                                end
                                
                                % Append accumulator c
                                in_block(end + 1) = c;
                                
                                % Sort and remove zeros
                                [~, sort_ord] = sort(abs(in_block), 'descend');
                                in_block = in_block(sort_ord);
                                in_block(in_block == 0) = [];
                                
                                % Call TC block
                                if ~isempty(in_block)
                                    d = Generic_BFMA_TC( ...
                                        in_block, NoExpBitsPrd, NoManBitsPrd, ...
                                        OutRoundMode, neab, stkbitenabled, ...
                                        NoManBitsOut, NoExpBitsOut, ...
                                        a_block, b_block, c, special_case,NoExpBitsIn);
                                else
                                    d = 0;
                                end
                                
                                c = d;     % recursive accumulation
                            
                            end % int pattern
                            
                        end     % end k loop
                        
                        D(m, n) = d;   % store result
                end % if Nan or Inf check
            end     % end n loop
        end         % end m loop
            
            % Toolbox not available or only single core
                    %disp('Running serial version...');
else % if M>2 && parallel_cores
%--------------------------------------------------------------------
%%      Serial simple for loops
%--------------------------------------------------------------------

            %% Recursive Addition of TC Results 
                for m = 1:M                             % rows of A
    
                    a = [reshape(A(m, :), 1, []),zeros(1,pad_size)];        % take m-th row of A
    
    

                for n = 1:N                         % columns of B
                    
                    % -------------------------------------------------------------
                    % Pad to make length an integral multiple of nfma
                    % -------------------------------------------------------------
                    b = [reshape(B(:, n), 1, []),zeros(1,pad_size)];    % take n-th column of B
                    
                    c = C(m, n);                    % element C(m, n)
                    
                    
                    
                    K = numel(a);
                    r = a .* b;                     % elementwise multiply
                    combined=[r,c];
                    if any(isnan(combined)) || any(isinf(combined))
                            if any(isnan(combined))
                                D(m,n) = NaN;           % NaN takes priority
                            elseif any(combined == Inf) && any(combined == -Inf)
                                D(m,n) = NaN;           % mixed +Inf and -Inf → ambiguous
                            elseif any(combined == Inf)
                                D(m,n) = Inf;           % only +Inf
                            elseif any(combined == -Inf)
                                D(m,n) = -Inf;          % only -Inf
                            end
                    
                    
                    % -------------------------------------------------------------
                    % Recursive accumulation via TC blocks
                    % -------------------------------------------------------------
                    else % if not nan, or inf
                    
                            for k = 1 : K / nfma
                                
                                % ---------------------------------------------------------
                                % Extract block of nfma elements
                                % ---------------------------------------------------------
                                in_block = r((k - 1) * nfma + 1 : k * nfma);
                                a_block =  a((k - 1) * nfma + 1 : k * nfma);
                                b_block =  b((k - 1) * nfma + 1 : k * nfma);
                                
                                % ---------------------------------------------------------
                                % Special case: interleaved pattern (H100 / B200 / H200)
                                % ---------------------------------------------------------
                                if inter_pattern
                                    
                                    %% ------------------ First interleaved block ------------------
                                    in_block_1 = in_block(seq1);
                                    in_block_2 = [in_block(seq2), 0];
                                    
                                    a_block_1 = a_block(seq1);
                                    b_block_1 = b_block(seq1);
                                    
                                    % Sort by magnitude
                                    [~, sort_ord] = sort(abs(in_block_1), 'descend');
                                    in_block_1 = in_block_1(sort_ord);
                                    
                                    % Special case: because c=0 here
                                    special_case = 1;
                                    
                                    % Remove zeros
                                    in_block_1(in_block_1 == 0) = [];
                                    
                                    % First half cycle
                                    if ~isempty(in_block_1)
                                        d1 = Generic_BFMA_TC( ...
                                            in_block_1, NoExpBitsPrd, NoManBitsPrd, ...
                                            OutRoundMode, neab, stkbitenabled, ...
                                            NoManBitsOut, NoExpBitsOut, ...
                                            a_block_1, b_block_1, 0, special_case,NoExpBitsIn);
                                    else
                                        d1 = 0;
                                    end
                                    
                                    %% ------------------ Second interleaved block ------------------
                                    
                                    % Re-check special-case flag
                                    if abs(d1) > abs(in_block_2)
                                        special_case = 0;
                                    else
                                        special_case = 1;
                                    end
                                    
                                    % Insert previous output into block
                                    in_block_2(end) = d1;
                                    
                                    % Sort again
                                    [~, sort_ord] = sort(abs(in_block_2), 'descend');
                                    in_block_2 = in_block_2(sort_ord);
                                    
                                    a_block_2 = a_block(seq2);
                                    b_block_2 = b_block(seq2);
                                    
                                    % Remove zeros
                                    in_block_2(in_block_2 == 0) = [];
                                    
                                    % Second half cycle
                                    if ~isempty(in_block_2)
                                        d2 = Generic_BFMA_TC( ...
                                            in_block_2, NoExpBitsPrd, NoManBitsPrd, ...
                                            OutRoundMode, neab, stkbitenabled, ...
                                            NoManBitsOut, NoExpBitsOut, ...
                                            a_block_2, b_block_2, d1, special_case,NoExpBitsIn);
                                    else
                                        d2 = 0;
                                    end
                                    
                                    %% ------------------ Final addition with C ------------------
                                    
                                    in_block_3 = [d2, c];
                                    
                                    [~, sort_ord] = sort(abs(in_block_3), 'descend');
                                    in_block_3 = in_block_3(sort_ord);
                                    
                                    in_block_3(in_block_3 == 0) = [];
                                    % neab=2, stikybitenabled=1, prd=output precsision
                                    if ~isempty(in_block_3)
                                        d = Generic_BFMA_TC( ...
                                            in_block_3, NoExpBitsOut, NoManBitsOut, ...
                                            cOutRoundMode, 2, 1, ...
                                            NoManBitsOut, NoExpBitsOut, ...
                                            [], [], 0, 0, NoExpBitsIn);
                                        
                                    else
                                        d = 0;
                                    end
                                    
                                    c = d;     % update accumulator
                                    
                                % ---------------------------------------------------------
                                % Non-interleaved pattern (standard TC path)
                                % ---------------------------------------------------------
                                else
                                    
                                    % Detect special-case
                                    if abs(c) > abs(in_block)
                                        special_case = 0;
                                    else
                                        special_case = 1;
                                    end
                                    
                                    % Append accumulator c
                                    in_block(end + 1) = c;
                                    
                                    % Sort and remove zeros
                                    [~, sort_ord] = sort(abs(in_block), 'descend');
                                    in_block = in_block(sort_ord);
                                    in_block(in_block == 0) = [];
                                    
                                    % Call TC block
                                    
                                    if ~isempty(in_block)
                                        d = Generic_BFMA_TC( ...
                                            in_block, NoExpBitsPrd, NoManBitsPrd, ...
                                            OutRoundMode, neab, stkbitenabled, ...
                                            NoManBitsOut, NoExpBitsOut, ...
                                            a_block, b_block, c, special_case,NoExpBitsIn);
                                    else
                                        d = 0;
                                    end
                                    
                                    c = d;     % recursive accumulation
                                end
                                
                            end     % end k loop
                            
                            D(m, n) = d;   % store result
                    end % if Nan or Inf check
                end     % end n loop
            end         % end m loop
end   % if M>2 && parallel_core
 
 
end






function fmt = fpformatinfo(fmtName)
%FPFORMATINFO  Return mantissa, exponent, and total bits for FP formats
%
%   fmt = fpformatinfo(fmtName)
%
%   Input:
%       fmtName : String specifying the floating-point format
%                 Options: 
%                   'binary64', 'binary32', 'binary16', 'bfloat16', 
%                   'tensorfloat32', 'fp8-e4m3', 'fp8-e5m2', 
%                   'fp6', 'fp4'
%
%   Output (struct):
%       fmt.totalBits   - Total number of bits
%       fmt.expBits     - Number of exponent bits
%       fmt.manBits     - Number of mantissa (fraction) bits
%       fmt.bias        - Exponent bias
%       fmt.hasImplicit - Whether the implicit leading 1 exists
%
%   Examples:
%       >> fpformatinfo('binary32')
%       ans =
%           totalBits: 32
%             expBits: 8
%             manBits: 23
%               bias: 127
%         hasImplicit: 1
%
%       >> fpformatinfo('fp8-e4m3')
%       ans =
%           totalBits: 8
%             expBits: 4
%             manBits: 3
%               bias: 7
%         hasImplicit: 1
%
% -------------------------------------------------------------------------
    switch lower(fmtName)
        case {'binary64','float64','double','fp64'}
            fmt.totalBits   = 64;
            fmt.expBits     = 11;
            fmt.manBits     = 52;
            fmt.hasImplicit = true;

        case {'binary32','float32','single','fp32'}
            fmt.totalBits   = 32;
            fmt.expBits     = 8;
            fmt.manBits     = 23;
            fmt.hasImplicit = true;

        case {'binary16','float16','half','fp16'}
            fmt.totalBits   = 16;
            fmt.expBits     = 5;
            fmt.manBits     = 10;
            fmt.hasImplicit = true;

        case {'bfloat16','bf16'}
            fmt.totalBits   = 16;
            fmt.expBits     = 8;
            fmt.manBits     = 7;
            fmt.hasImplicit = true;

        case {'tensorfloat32','tf32'}
            % NVIDIA TF32: FP32 exponent range, 10 mantissa bits
            fmt.totalBits   = 19; 
            fmt.expBits     = 8;
            fmt.manBits     = 10;
            fmt.hasImplicit = true;

        case {'fp8-e4m3','e4m3'}
            fmt.totalBits   = 8;
            fmt.expBits     = 4;
            fmt.manBits     = 3;
            fmt.hasImplicit = true;

        case {'fp8-e5m2','e5m2'}
            fmt.totalBits   = 8;
            fmt.expBits     = 5;
            fmt.manBits     = 2;
            fmt.hasImplicit = true;

        case {'fp6','e3m2'}
            fmt.totalBits   = 6;
            fmt.expBits     = 3;
            fmt.manBits     = 2;
            fmt.hasImplicit = true;

        case {'fp4','e2m1'}
            fmt.totalBits   = 4;
            fmt.expBits     = 2;
            fmt.manBits     = 1;
            fmt.hasImplicit = true;

        otherwise
            warning('Use CPFloat library with custom input precision and add it here as a case and then it will work')
            error('Unknown format "%s". Supported: binary64, binary32, binary16, bfloat16, tensorfloat32, fp8-e4m3, fp8-e5m2, fp6, fp4', fmtName);
            
    end

    % Compute exponent bias (IEEE rule: 2^(k-1)-1)
    fmt.bias = 2^(fmt.expBits-1) - 1;

end





