% matmul_test.m
%
% Requirements:
%   CPFloat (https://github.com/north-numerical-computing/cpfloat/).
%
% References:
%   [1] T. Mary and M. Mikaitis.
%       Error Analysis of Matrix Multiplication with Narrow Range
%       Floating-Point Arithmetic. hal-04671474. Aug. 2024.

[~, options] = cpfloat([], inopt);
% Grab various parameters of the format.
t    = options.params(1);
emin = options.params(2);
emax = options.params(3);
% Set up some useful quantities used in the paper.
u = 2^-t;
fmin = 2^emin;
fmax = 2^emax*(2-2*u);

[~, options] = cpfloat([], outopt);
% Grab various parameters of the format.
T    = options.params(1);
Emin = options.params(2);
Emax = options.params(3);
% Set up some useful quantities used in the paper.
U = 2^(-T);
Fmin = 2^Emin;
Fmax = 2^Emax*(2-2*U);

% Matrix dimensions: A is m x n, B is n x q
m = 10;
q = 10;
nlist = floor(logspace(1,6,20));
% Matrix elements: uniformly distributed logarithms in [-l, l].
l = 10;

i = 0;
for n = nlist
    i=i+1;
    
    % Matrix elements: uniformly distributed logarithms in [-l, l].
    A = (10.^(rand(m,n)*2*l-l));
    B = (10.^(rand(n,q)*2*l-l));
    % Random sign + or - with equal probability
    sA = randi(2,m,n)*2-3;
    sB = randi(2,n,q)*2-3;
    A = A.*sA;
    B = B.*sB;
    
    % Compute a reference result in binary64.
    Ctrue = A*B;
    
    % Compute diagonal scaling matrices L and M such that
    % the elements of L*A and B*M are at most theta.
    theta = min(fmax, sqrt(Fmax/n));
    L = previous_pow2(theta./max(abs(A),[],2));
    M = previous_pow2(theta./max(abs(B)));
    Linv = 1./L;
    Minv = 1./M;
    
    % Round L*A and B*M to the input format.
    temp1 = L.*A;
    temp2 = B.*M;
    LA{1} = cpfloat(temp1, inopt);
    BM{1} = cpfloat(temp2, inopt);

    % Split into p words
    for j = 2:p
        temp1 = temp1 - LA{j-1}*u^(j-2);
        temp2 = temp2 - BM{j-1}*u^(j-2);
        LA{j} = cpfloat(temp1/u^(j-1), inopt); 
        BM{j} = cpfloat(temp2/u^(j-1), inopt);
    end
    
    % Compute LA*BM in the accumulation format.
    LABM = matmul(LA, BM, model, p, u, inopt, outopt);
    
    % Scale LABM back to obtain C.
    C = Linv.*LABM.*Minv;
    
    % Compute the error
    err(i) = norm(C-Ctrue,'inf')/norm(A,'inf')/norm(B,'inf');
    
    % Bound (3.26)
    bound(i) = 2*u + n*U + 4*n^2*fmin/theta + 4*n^2*Fmin/theta^2;
    % Same bound but without dependency on n, which is quite
    % pessimistic.
    %bound(i) = 2*u + U + 4*fmin/theta + 4*Fmin/theta^2;
    %bound_nrl(i) = 2*u + U;
end

% Output various results to .dat files.
filename = strcat('./data/matmul_test_', inopt.format,...
    '_', outopt.format, '_words_', num2str(p), '_model_', model, '.dat');
fileID = fopen(filename, 'w');
fprintf(fileID, ...
    ['n error bound \n']);
for j=1:length(nlist)
    fprintf(fileID,'%d %e %e \n', ...
        nlist(j), err(j), bound(j));
end

function y = previous_pow2(x)
% Replace elements of x by the immediately inferior power of two.
y = 2.^floor(log2(x));
end

function C = matmul(A, B, model, p, u, inopt, outopt)
  C = zeros(size(A{1},1), size(B{1},2));
  for j=1:p
      for k=1:p
          if (j+k-2 < p)
              if strcmp(model, 'v100')
                  C = V100TC(u^(j+k-2), A{j}, B{k}, 1, C, outopt.format);
              elseif strcmp(model, 'a100')
                  C = A100TC(u^(j+k-2), A{j}, B{k}, 1, C,...
                  inopt.format, outopt.format);
              elseif strcmp(model, 'b200')
                  C = B200TC(u^(j+k-2), A{j}, B{k}, 1, C, ...
                      inopt.format, outopt.format);
              elseif strcmp(model, 'b200rn')
                  C = B200TCRN(u^(j+k-2), A{j}, B{k}, 1, C, ...
                      inopt.format, outopt.format);
              elseif strcmp(model, 'l40s')
                  C = L40STC(u^(j+k-2), A{j}, B{k}, 1, C, ...
                      inopt.format, outopt.format);
              end
          end
      end
  end
end