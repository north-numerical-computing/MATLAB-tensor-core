warning('off', 'all');
clear all
alpha=1;beta=1;
Model_GPU_error=0;
GPUModels={'Ada','H100','H200','A100','V100','B200'};
outputformatd='fp32';
outputformatc='fp32'; % c was considered always in fp32, if needs in fp16, roundtonearest was used in CUDA

addpath tools\

%% trimmed data
baseDir = '';
sub='';

for gpunumber=1:numel(GPUModels)
GPUModel=GPUModels{gpunumber};
switch GPUModel
    case 'V100'
        inputformats={'fp16'};
        Ks=4;
    case {'A100','A2','A30'}
        inputformats={'fp16','bf16','tf32'};
        Ks=[8,8,4];
    case {'H100','B200','H200'}
        inputformats={'fp16','bf16','tf32','E5M2','E4M3'};
        Ks=[16,16,4,32,32];
    case {'Ada','L40S'}
        inputformats={'fp16','bf16','tf32','E5M2','E4M3'};
        Ks=[8,8,4,32,32];
    case{'Cust'}
        inputformats={'bf16'};
        Ks=16;
    otherwise
end
%
for inp_for_number=1:numel(inputformats)
    K=Ks(inp_for_number);
inputformat=inputformats{inp_for_number};
switch inputformat
    case 'fp16'
        outputformatds={'fp16','fp32'};
    otherwise
        outputformatds={'fp32'};
end

for out_for_number=1:numel(outputformatds)
outputformatd=outputformatds{out_for_number};

folders = inputformat;
fullPathA = [baseDir,GPUModel,'\' folders,'\',sub,'a_',GPUModel,'_',inputformat,'.txt'];
fullPathB = [baseDir,GPUModel,'\' folders,'\',sub,'b_',GPUModel,'_',inputformat,'.txt'];
fullPathC = [baseDir,GPUModel,'\' folders,'\',sub,'c_',GPUModel,'_',outputformatc,'.txt'];
fullPathD = [baseDir,GPUModel,'\' folders,'\',sub,'d_',GPUModel,'_',outputformatd,'.txt'];
A = readHexFloatFile(fullPathA);
A=reshape(A,K,numel(A)/K);
A=transpose(A);
B = readHexFloatFile(fullPathB);
B=reshape(B,K,numel(B)/K);
B=transpose(B);
isFP8 = ismember(lower(inputformat),{'e5m2','e4m3','fp8-e4m3','fp8-e5m2'});
if ~(ismember(GPUModel, {'H100','H200'}) && isFP8)
    C=readIeeeFloatsFromFile(fullPathC);
else
    C=zeros(size(C));
    % D= AB+D where D is zero for fp8 with WGMMA
end
DGPU = readIeeeFloatsFromFile(fullPathD);

%% input and output formats
clear dm
clc
enssize=numel(C(:,1));
range=1:enssize;
clear dm
for count=range
a=A(count,:);
b=B(count,:);
b=transpose(b);
c=C(count);
% change the function name here
switch GPUModel
    case 'V100'
    [dm(count)]=V100TC(1,a,b,1,c,outputformatd);
    case {'B200'}
    [dm(count)]=B200TC(1,a,b,1,c,inputformat,outputformatd);    
    case {'H200','H100'}    
    [dm(count)]=H100TC(1,a,b,1,c,inputformat,outputformatd);
    case {'A100','A2','A30'}
    [dm(count)]=A100TC(1,a,b,1,c,inputformat,outputformatd);
    case {'Ada','L40S'}
    [dm(count)]=AdaTC(1,a,b,1,c,inputformat,outputformatd);
    otherwise
end
count
end

dif=[dm(range)'-DGPU(range)];
Model_GPU_error=sum(abs(dif))+Model_GPU_error;
if Model_GPU_error~=0

    error('Error occurred in model %s during simulation', model);

end

end
end % input_format_number
end
Model_GPU_error
