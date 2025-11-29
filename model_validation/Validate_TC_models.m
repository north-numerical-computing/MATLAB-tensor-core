clear all
GPUModel='V100';
inputformat='fp16';
outputformatd='fp16';
switch GPUModel
    case 'V100'
        
        K=4; % vector size generated in the text file
       
        % A100 
    case {'A100','A2','A30','A40'}
        switch inputformat
            case {'fp16','bf16','bfloat16'}
                K=8;
            case 'tf32'
                K=4;
            otherwise
                K=8;
             
        end
        % Ada L40S case
    case {'Ada','L40S'}
        switch inputformat
            case {'fp16','bf16','bfloat16'}
                K=8;
            case {'tf32','tensorfloat32'}
                K=4;
            case {'E5M2','E4M3','fp8-e4m3','fp8-e5m2'}
                K=32;
            otherwise
                K=8;
              
        end
        % B200 case
    case {'B200','H200','H100'}
            switch inputformat
            case {'fp16','bf16','bfloat16'}
                K=16;
            case {'tf32','tensorfloat32'}
                K=4;
            case {'E5M2','E4M3','fp8-e4m3','fp8-e5m2'}
                K=32;
            otherwise
                K=16;
              
            end
    otherwise
end

outputformatc='fp32'; % c was considered always in fp32, if needs in fp16, roundtonearest was used in CUDA

%
baseDir = '';
folders = inputformat;
fullPathA = [baseDir,GPUModel,'\' folders,'\a_',GPUModel,'_',inputformat,'.txt'];
fullPathB = [baseDir,GPUModel,'\' folders,'\b_',GPUModel,'_',inputformat,'.txt'];
fullPathC = [baseDir,GPUModel,'\' folders,'\c_',GPUModel,'_',outputformatc,'.txt'];
fullPathD = [baseDir,GPUModel,'\' folders,'\d_',GPUModel,'_',outputformatd,'.txt'];
A = readHexFloatFile(fullPathA);
A=reshape(A,K,numel(A)/K);
A=transpose(A);
B = readHexFloatFile(fullPathB);
B=reshape(B,K,numel(B)/K);
B=transpose(B);
C=readIeeeFloatsFromFile(fullPathC);
DGPU = readIeeeFloatsFromFile(fullPathD);

%% input and output formats
clear dm
clc
range=1:10000;
for i=range
a=A(i,:);
b=B(i,:);
b=transpose(b);
c=C(i);
% change the function name here
[dm(i)]=V100TC(a,b,c,outputformatd);
i
end

dif=[dm(range)'-DGPU(range)];
sum(abs(dif))

