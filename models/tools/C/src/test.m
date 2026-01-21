def_params.fma = 16;          % Fused multiply-add (FMA) size
def_params.neab = 2;          % TC extra alignment bits
def_params.frmode = 'rz';     % TC final rounding mode
def_params.stkbitenabled = 0;
def_params.inter_pattern=0;

for i=1:10^4
    A = rand(16,16*10^2) - 0.5;
    B = rand(16*10^2,16) - 0.5;
    C = double(single(rand(16,16)));
    
    A1=A;
    B1=B;
    C1=C;
    C1=C*1;
    Cmodel = gemm(1, A, B, 1, C, 'binary16', 'binary32', def_params);
    Mmodel = GEMM(1, A1, B1, 1, C1, 'binary16', 'binary32', def_params);

    % num2hex(single(Cmodel))
    % num2hex(single(Mmodel))

    if (Cmodel ~= Mmodel)
        fprintf("%.30f %.30f %.30f %.30f \n", A);
        fprintf("%.30f %.30f %.30f %.30f \n", B');
        fprintf("%.30f \n", C);
    end
end