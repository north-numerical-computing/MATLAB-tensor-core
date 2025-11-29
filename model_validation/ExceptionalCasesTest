%% ========================================================================
%   Testbench for CustomTC Tensor Core Model – Exceptional Case Handling
%   This script validates Infinity, -Infinity, and NaN propagation behaviour
% ========================================================================

addpath ../models/tools/

%% ---------------------------- Model Configuration -----------------------
params.fma           = 16;         % Fused multiply-add (FMA) size
params.neab          = 1;         % Extra alignment bits
params.frmode        = 'rne';      % Final rounding mode
params.stkbitenabled = 0;         % Enable stacked bits (disabled)
params.inter_pattern = 0;         % Interleaving pattern flag

%% ---------------------------- Global Host Operands ---------------------
global hA hB hC
hA = zeros(1,16);
hB = zeros(16,1);
hC = zeros(1,1);

%% ---------------------------- Host Reset Function -----------------------
function host_reset()
    global hA hB hC
    hA = zeros(1,16);
    hB = zeros(16,1);
    hC = zeros(1,1);
end

%% ---------------------------- Input Formats -----------------------------
inFormat1 = 'bf16';
inFormat2 = 'tf32';

inputOpts.format = inFormat1;
inputOpts.round  = 4;

% Uncomment for TF32
inputOpts.format = inFormat2;
inputOpts.round  = 4;

outputFormat = 'fp32';


%% ========================================================================
% 1. Overflow in accumulation: c + a1*b1 results in +Inf
% ========================================================================
host_reset();
hC(1) = typecast(uint32(hex2dec('7F000000')), 'single');   % ~2^127
hA(1) = cpfloat(hC(1), inputOpts);
hB(1) = 1;

result = CustomTC(1, hA, hB, 1, hC, inFormat1, outputFormat, params);

if isinf(result) && result > 0
    disp('Test 1 PASSED: Accumulation overflow correctly produced +Inf.');
end

%% ========================================================================
% 2. Multiplication of +Inf × -Inf → -Inf (no NaN expected)
% ========================================================================
host_reset();
hA(1) = cpfloat(typecast(uint32(hex2dec('7F800000')), 'single'), inputOpts); % +Inf
hB(1) = cpfloat(typecast(uint32(hex2dec('FF800000')), 'single'), inputOpts); % -Inf

result = CustomTC(1, hA, hB, 1, hC, inFormat1, outputFormat, params);

if isinf(result) && result < 0
    disp('Test 2 PASSED: (+Inf × -Inf) correctly returned -Inf.');
end

%% ========================================================================
% 3. Addition of +Inf and -Inf → NaN
% ========================================================================
host_reset();
hA(1) = -Inf;
hB(1) = 1;
hA(2) = +Inf;
hB(2) = 1;

result = CustomTC(1, hA, hB, 1, hC, inFormat1, outputFormat, params);

if isnan(result)
    disp('Test 3 PASSED: (+Inf + -Inf) correctly returned NaN.');
end

%% ---------------------- Half-Precision Tests ---------------------------
inFormatFP16  = 'fp16';
outFormatFP16 = 'fp16';
inputOptsFP16.format = inFormatFP16;
inputOptsFP16.round  = 4;

% 4. Overflow to +Inf in fp16 accumulation
host_reset();
hC(1) = typecast(uint32(hex2dec('78000000')), 'single');
hA(1) = cpfloat(hC(1), inputOptsFP16);
hA(2) = cpfloat(hC(1), inputOptsFP16);
hC(1) = 0;
hB(1:2) = 1;

result = CustomTC(1, hA, hB, 1, hC, inFormatFP16, outFormatFP16, params);

if isinf(result) && result > 0
    disp('Test 4 PASSED: fp16 accumulation correctly overflowed to +Inf.');
end

% 5. Overflow to -Inf in fp16 accumulation
host_reset();
hC(1) = typecast(uint32(hex2dec('F8000000')), 'single');
hA(1) = cpfloat(hC(1), inputOptsFP16);
hA(2) = cpfloat(hC(1), inputOptsFP16);
hC(1) = 0;
hB(1:2) = 1;

result = CustomTC(1, hA, hB, 1, hC, inFormatFP16, outFormatFP16, params);

if isinf(result) && result < 0
    disp('Test 5 PASSED: fp16 accumulation correctly overflowed to -Inf.');
end

%% ---------------------- FP8 Exceptional Behaviour Tests ----------------
inFormatFP8  = 'fp8-e5m2';
outFormatFP8 = 'fp16';
inputOptsFP8.format = inFormatFP8;
inputOptsFP8.round  = 4;

% 6. Overflow to +Inf for fp8 input
host_reset();
hC(1) = typecast(uint32(hex2dec('78000000')), 'single');
hA(1) = cpfloat(hC(1), inputOptsFP8);
hA(2) = cpfloat(hC(1), inputOptsFP8);
hC(1) = 0;
hB(1:2) = 1;

result = CustomTC(1, hA, hB, 1, hC, inFormatFP8, outFormatFP8, params);

if isinf(result) && result > 0
    disp('Test 6 PASSED: fp8 input accumulation correctly overflowed to +Inf.');
end

% 7. Overflow to -Inf for fp8 input
host_reset();
hC(1) = typecast(uint32(hex2dec('78000000')), 'single');
hA(1) = cpfloat(hC(1), inputOptsFP8);
hA(2) = cpfloat(hC(1), inputOptsFP8);
hC(1) = 0;
hB(1:2) = -1;

result = CustomTC(1, hA, hB, 1, hC, inFormatFP8, outFormatFP8, params);

if isinf(result) && result < 0
    disp('Test 7 PASSED: fp8 input accumulation correctly overflowed to -Inf.');
end
