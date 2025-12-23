MATLAB Tensor Core models
--

[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=north-numerical-computing/MATLAB-tensor-core)

## Overview

This repository provides accurate tensor core models written in MATLAB. It also includes parts of the model validation data which is used to refine the models as shown in [1].

The [models](models/) directory contains the MATLAB models of tensor cores of several NVIDIA GPUs, all of which are build on the parameterised model in [Generic_BFMA_TC.m](models/tools/Generic_BFMA_TC.m). For example the [B200TC.m](models/B200TC.m) models the General Matrix Multiply (GEMM) based on the accurate model of a tensor core in the NVIDIA Blackwell B200 GPUs. In the current version of the toolbox, the models take matrices and input and output floating-point formats as inputs and multiply the matrices by using a recursive summation algorithm to accummulate the results of several tensor core invocations.

The initial analysis of the behaviour of GPU tensor cores is performed with the code available at [IEEE_HPEC2025_block_FMA_tests](https://github.com/faiziktk/IEEE_HPEC2025_block_FMA_tests).
It is based on the generalised testing methodology [2] which determines the following features of hardware computing mixed-precision inner products:

* Support for subnormal numbers
* Presence of extra bits for significand alignment in multi-term addition
* Availability of extra carry bits
* Normalization patterns in multi-term floating-point addition
* Supported rounding modes
* Effective FMA size (i.e., number of terms accumulated before a single normalization)

The [model_validation](data/model_validation) contains part of the model validation data that was used in [1] to refine the models and verify the bit-accurate behaviour against the corresponding GPUs. Full-sized experiments and data is not stored in this repository but is available on request.

The [experiments](experiments/) directory contains various experiments with some of the [models](models/) that were performed to plot the results in [1]. These can serve as examples on how to utilise the models.

## Dependencies and installation

1. Set up the custom precision floating-point format simulator [CPFloat](https://github.com/north-numerical-computing/cpfloat).
2. Add [models/](models/) to the MATLAB search path.
3. Add [models/tools](models/tools) to the MATLAB search path.

## Example: Using in-built models

The following example rounds two matrices to fp16 and multiplies them using the model of the B200 tensor core.
Note that B200TC computes the GEMM, with alpha and beta scale factors set to 1.

```
>> inopts.format = 'binary16';
>> outopts.format = 'binary32';
>> A = cpfloat(rand(4,4), inopts);
>> B = cpfloat(rand(4,4), inopts);
>> B200TC(1, A, B, 1, 0, inopts.format, outopts.format)

ans =

   0.995566666126251   1.208170533180237   1.368334889411926   1.017799258232117
   0.991239666938782   1.084852933883667   1.350871562957764   1.328557014465332
   1.190854787826538   1.693876862525940   1.763551592826843   1.278026223182678
   0.901759386062622   1.838499188423157   1.608222723007202   1.265371918678284
  ```

The following example uses an 8-bit floating-point format as the input format in the B200 tensor core model.

```
>> inopts.format = 'fp8-e4m3';
>> A = cpfloat(rand(4,4), inopts);
>> B = cpfloat(rand(4,4), inopts);
>> B200TC(1, A, B, 1, 0, inopts.format, outopts.format)

ans =

   0.390136718750000   0.589843750000000   0.625976562500000   0.748046875000000
   1.180175781250000   1.117187500000000   1.220703125000000   1.935546875000000
   1.267822265625000   0.752929687500000   0.867187500000000   1.813476562500000
   1.007812500000000   1.242187500000000   1.395996093750000   1.740234375000000
```

## Example: Setting up the NVIDIA B200 model

While the B200 tensor core model comes with this toolbox, below is a minimal example for setting it up. The input matrices are assumed to be rounded to the appropriate formats with CPFloat. The model in [B200TC.m](models/B200TC.m) provides a more detailed set up that changes the parameters of a generalised model based on all possible input/output format combinations.

```
% Default structures assuming fp16 in and fp32 output
def_params.fma    = 16;      % Fused multiply-add (FMA) size
def_params.neab   = 2;       % TC extra alignment bits
def_params.frmode = 'rz';    % TC final rounding mode
def_params.inter_pattern=1;  % Interleave two 16-element vectors

D = GEMM(alpha, A, B, beta, C, informat, outformat, def_params);
```

## References

[1] F. A. Khattak and M. Mikaitis, [Accurate Models of NVIDIA Tensor Cores](https://arxiv.org/abs/2512.07004). arXiv:2512.07004 [cs.MS]. Dec. 2025.<br>
[2] F. A. Khattak and M. Mikaitis, [Generalized Methodology for Determining Numerical Features of Hardware Floating-Point Matrix Multipliers: Part I](https://ieeexplore.ieee.org/abstract/document/11196657). 2025 IEEE High Performance Extreme Computing Conference (HPEC). Sep. 2025.<br>
