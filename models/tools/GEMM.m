function D=GEMM(alpha, A, B, beta, C, informat, outformat, params)

    warning('off', 'all')
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
    params.NoManBitsOut = fmt.manBits;
    params.NoExpBitsOut = fmt.expBits;
    fmt = fpformatinfo(informat);
    params.NoExpBitsIn = fmt.expBits;
    params.NoManBitsIn = fmt.manBits;

    %% --------------------------------------------------------
    % Simulation Model Params
    % ---------------------------------------------------------
    nfma=params.fma;
    % Special case: Ada / L40S where number of alignment bits is negative for fp8
    if params.neab < 0
            % Special case of Ada/L40S
            params.NoManBitsOut = params.NoManBitsOut + params.neab;
        
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
    


if M>2 && exist('ver','file') && ~isempty(ver('parallel')) && exist('feature') && feature('numcores') > 1
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
                        if  any(isnan(combined)) || any(isinf(combined))
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
                                a_block =  a((k - 1) * nfma + 1 : k * nfma);
                                b_block =  b((k - 1) * nfma + 1 : k * nfma);
                                
                                % Call TC block
                                
                                    d = Generic_BFMA_TC(a_block, b_block, c, params);
                                
                                
                                c = d;     % recursive accumulation
                            
                            
                            
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
                                
                                a_block =  a((k - 1) * nfma + 1 : k * nfma);
                                b_block =  b((k - 1) * nfma + 1 : k * nfma);
                                d = Generic_BFMA_TC(a_block, b_block, c, params);
                                c = d;     % recursive accumulation
                               
                                
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

        case {'tensorfloat32','tf32','xf32'} % xf32 from AMD Guys
            % NVIDIA TF32: FP32 exponent range, 10 mantissa bits
            fmt.totalBits   = 19; 
            fmt.expBits     = 8;
            fmt.manBits     = 10;
            fmt.hasImplicit = true;

        case {'fp8-e4m3','e4m3','fp8'} % fp8 from AMD Guys
            fmt.totalBits   = 8;
            fmt.expBits     = 4;
            fmt.manBits     = 3;
            fmt.hasImplicit = true;

        case {'fp8-e5m2','e5m2','bf8'}  % bf8 solely for e5m2, from AMD Guys
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

