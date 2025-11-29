% experiments.m
%
% Requirements:
%   CPFloat (https://github.com/north-numerical-computing/cpfloat/).
%
% References:
%   [1] T. Mary and M. Mikaitis.
%       Error Analysis of Matrix Multiplication with Narrow Range
%       Floating-Point Arithmetic. hal-04671474. Aug. 2024.

clear all;

seed = 1;
rng(seed);

inopt.format='binary16';
outopt.format='binary32';

model = 'v100';
p = 1;
matmul_test;
p = 2;
matmul_test;
p = 3;
matmul_test;

rng(seed);

model = 'a100';
p = 1;
matmul_test;
p = 2;
matmul_test;
p = 3;
matmul_test;

rng(seed);

model = 'b200';
p = 1;
matmul_test;
p = 2;
matmul_test;
p = 3;
matmul_test;

rng(seed);

model = 'b200rn';
p = 1;
matmul_test;
p = 2;
matmul_test;
p = 3;
matmul_test;

inopt.format='bfloat16';
outopt.format='binary32';

rng(seed);

model = 'a100';
p = 1;
matmul_test;
p = 2;
matmul_test;
p = 3;
matmul_test;

rng(seed);

model = 'b200';
p = 1;
matmul_test;
p = 2;
matmul_test;
p = 3;
matmul_test;

rng(seed);

model = 'b200rn';
p = 1;
matmul_test;
p = 2;
matmul_test;
p = 3;
matmul_test;

inopt.format='fp8-e5m2';
outopt.format='binary32';

rng(seed);

model = 'l40s';
p = 1;
matmul_test;
p = 4;
matmul_test;
p = 6;
matmul_test;

rng(seed);

model = 'b200';
p = 1;
matmul_test;
p = 4;
matmul_test;
p = 6;
matmul_test;