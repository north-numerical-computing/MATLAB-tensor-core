#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <bitset>
#include <iomanip>
#include <cuda_bf16.h>
//#include <unistd.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include<cuda_fp8.h>
#include<cuda_fp16.hpp>

using namespace std;

#define K 32


/*--------- Parameter Selection by User------------------------------*/

//1. Choose intype 
/* choice one, uncomment all three lines*/
//#define TYPEC __NV_E5M2
//#define FMT e5m2  // or e4m3
//#define IN_FORMAT_E5M2

/*choice two, uncomment all three lines */
#define TYPEC __NV_E4M3
#define FMT e4m3  // or e4m3
#define IN_FORMAT_E4M3


//3. Choose GPUmodel
//#define B200
//#define L40S
//#define H200
//#define H100
#define Ada

/*-------------- End of Parameter of Selection by User----------------------*/







#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)


void printIEEE754(float f) {
    // Reinterpret float bits as 32-bit unsigned integer
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);

    // Print binary with spacing: 1 | 8 | 23 (sign | exponent | fraction)
    std::bitset<32> b(bits);
    //std::cout << "Float: " << f << "\n";
    std::cout << std::setprecision(std::numeric_limits<float>::digits10)
        << std::fixed
        << "Float: " << f << "\n";
    std::cout << "IEEE 754 (binary): ";
    std::cout << b[31] << " | ";                   // Sign bit
    for (int i = 30; i >= 23; --i)                 // Exponent bits
        std::cout << b[i];
    std::cout << " | ";
    for (int i = 22; i >= 0; --i)                  // Fraction bits
        std::cout << b[i];
    std::cout << std::endl;


}

// Function to print the bits of a __half
void print_half_bits(__half h) {
    // Reinterpret __half as 16-bit integer
    uint16_t bits = *reinterpret_cast<uint16_t*>(&h);

    // Print as binary
    std::bitset<16> b(bits);
    std::cout << b << std::endl;
}




void printFractionIEEE(float f)
{
    // Reinterpret float bits as 32-bit unsigned integer
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);

    // Print binary with spacing: 1 | 8 | 23 (sign | exponent | fraction)
    std::bitset<32> b(bits);
    //std::cout << "Float: " << f << "\n";
    std::cout << std::setprecision(std::numeric_limits<float>::digits10)
        << std::fixed
        << "Float: " << f << "\n";
    std::cout << " Full Fraction 1.";
    for (int i = 22; i >= 0; --i)                  // Fraction bits
        std::cout << b[i];
    std::cout << std::endl;


}


__global__ void mma_fp16_acc_fp32(float* out, const float* v, const float* u, float cc) {



    __nv_fp8_storage_t a_fp8[K / 2] = { 0 }; //16 elements, makes 8 halfs, 4 float
    __nv_fp8_storage_t b_fp8[K / 4] = { 0 };

    float c[4] = { 0, 0, 0, 0 };
    float d[4] = { 0, 0, 0, 0 };


    int window;
    if (K == 16)
    {
        window = 4;
    }
    else
    {
        window = 8;

    }
    // assignment of input to mma from two vectors a and b
    int tid = threadIdx.x; // warp thread index
    switch (tid) {
    case 0:
        // index 0 to 3
        c[0] = cc;
        for (int i = 0; i < 4; i++)
        {
            a_fp8[i] = __nv_cvt_float_to_fp8(v[i], __NV_SATFINITE, TYPEC);
            b_fp8[i] = __nv_cvt_float_to_fp8(u[i], __NV_SATFINITE, TYPEC);

        }
        if (K == 32)
        {
            for (int i = 0; i < 4; i++)
            {
                a_fp8[i + 8] = __nv_cvt_float_to_fp8(v[i + 16], __NV_SATFINITE, TYPEC);
                b_fp8[i + 4] = __nv_cvt_float_to_fp8(u[i + 16], __NV_SATFINITE, TYPEC);
            }
        }
        break;
    case 1:
        c[0] = 0;
        // index 0 to 3
        for (int i = 0; i < 4; i++)
        {
            a_fp8[i] = __nv_cvt_float_to_fp8(v[4 * tid + i], __NV_SATFINITE, TYPEC);
            b_fp8[i] = __nv_cvt_float_to_fp8(u[4 * tid + i], __NV_SATFINITE, TYPEC);
        }

        // index 8 to 11 for a, 4 to 7 for b 
        if (K == 32)
        {
            for (int i = 0; i < 4; i++)
            {
                a_fp8[i + 8] = __nv_cvt_float_to_fp8(v[tid * 4 + i + 16], __NV_SATFINITE, TYPEC);
                b_fp8[i + 4] = __nv_cvt_float_to_fp8(u[tid * 4 + i + 16], __NV_SATFINITE, TYPEC);
            }
        }

        break;
    case 2:

        for (int i = 0; i < 4; i++)
        {
            a_fp8[i] = __nv_cvt_float_to_fp8(v[4 * tid + i], __NV_SATFINITE, TYPEC);
            b_fp8[i] = __nv_cvt_float_to_fp8(u[4 * tid + i], __NV_SATFINITE, TYPEC);
        }
        // index 8 to 11 for a,
        if (K == 32)
        {
            for (int i = 0; i < 4; i++)
            {
                a_fp8[i + 8] = __nv_cvt_float_to_fp8(v[tid * 4 + i + 16], __NV_SATFINITE, TYPEC);
                b_fp8[i + 4] = __nv_cvt_float_to_fp8(u[tid * 4 + i + 16], __NV_SATFINITE, TYPEC);
            }
        }
        break;
    case 3:


        for (int i = 0; i < 4; i++)
        {
            a_fp8[i] = __nv_cvt_float_to_fp8(v[4 * tid + i], __NV_SATFINITE, TYPEC);
            b_fp8[i] = __nv_cvt_float_to_fp8(u[4 * tid + i], __NV_SATFINITE, TYPEC);
        }
        // index 8 to 11 for a, 
        if (K == 32)
        {
            for (int i = 0; i < 4; i++)
            {
                a_fp8[i + 8] = __nv_cvt_float_to_fp8(v[tid * 4 + i + 16], __NV_SATFINITE, TYPEC);
                b_fp8[i + 4] = __nv_cvt_float_to_fp8(u[tid * 4 + i + 16], __NV_SATFINITE, TYPEC);
            }
        }
        break;
    default:
        for (int i = 0; i < 8; i++)
        {
            a_fp8[i] = __nv_cvt_float_to_fp8(0, __NV_SATFINITE, TYPEC);
        }

        for (int i = 0; i < window; i++)
        {
            b_fp8[i] = __nv_cvt_float_to_fp8(0, __NV_SATFINITE, TYPEC);
        }
        break;
    }



    // Convert FP8 to half precision
    unsigned const* A = reinterpret_cast<unsigned const*>(&a_fp8);
    unsigned const* B = reinterpret_cast<unsigned const*>(&b_fp8);
    float const* C = reinterpret_cast<float const*>(&c);
    float* D = reinterpret_cast<float*>(&d);


#if K == 32
    asm(
        "mma.sync.aligned.m16n8k32.row.col.f32." STR(FMT) "." STR(FMT) ".f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        :
        "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
    );

#elif K == 16
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32." STR(FMT) "." STR(FMT) ".f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
    );
#endif  

    /*if (threadIdx.x == 0)
    {
        printf("Thread index=0 value is=%f\n",D[0]);
    }*/
    memcpy(out + threadIdx.x * 2, D, 8);
    memcpy(out + 8 * 8 + threadIdx.x * 2, D + 2, 8);

}

void resetarrays(float arr1[], float arr2[], int size) {
    for (int i = 0; i < size; i++) {
        arr1[i] = 0.0f;
        arr2[i] = 0.0f;
    }
}


void writeFloatBitsToFile(const string& filename, float value) {
    ofstream outFile(filename, ios::app);
    if (!outFile) {
        cerr << "Error opening file!" << endl;
        return;
    }

    uint32_t bits;
    memcpy(&bits, &value, sizeof(float));

    outFile << bitset<32>(bits) << endl;  // write as binary string
    outFile.close();
}

int main() {
    const int N = K;
    float h_b[N] = { 1 };
    float h_a[N] = { 1 };
    float c;
    float* d_b;
    float* d_a;
    float* d_c2;
    float* h_C = (float*)malloc(16 * 8 * sizeof(float));
    float* d_C;


    // device specific setting
    
    //cudaMalloc(&d_c2, *sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_C, 16 * 8 * sizeof(float));

    //**********************************************************************
  
    string intype = "E4M3";// change this to E4M3 for the second format
    string GPUmodel = "H200";
    string outtype = "fp32";
    string path;


#ifdef IN_FORMAT_E5M2
    intype = "E5M2";
#endif // 
#ifdef IN_FORMAT_E4M3
    intype = "E4M3";
#endif // 


    // Models pick from macros
#if defined(B200)
    GPUmodel = "B200";
#elif defined(L40S)
    GPUmodel = "L40S";
#elif defined(H200)
    GPUmodel = "H200";
#elif defined(H100)
    GPUmodel = "H100";
#elif defined(Ada)
    GPUmodel = "Ada";
#else
    GPUmodel = "Unknown";
#endif






    ///////////////////////////////////////////////////////////////////////////////
// File path: set it here    
        path = ""; // uncomment for GPU results
        path = path + GPUmodel + "/" + intype + "/";

    string outfilename = path + "d_" + GPUmodel + "_" + outtype + ".txt";

    ifstream inFileA(path + "a_" + GPUmodel + "_" + intype + ".txt"); // a
    ifstream inFileB(path + "b_" + GPUmodel + "_" + intype + ".txt"); // b
    ifstream inFileC(path + "c_" + GPUmodel + "_" + outtype + ".txt");

    if (!inFileA || !inFileB || !inFileC) {
        cerr << "Error opening one or more files!" << endl;
        return 1;
    }

    const int k = N;  // elements per row
    float h_aa[k], h_bb[k];
    float h_cc[1];     // single element from c file
    int rowsToRead = 10;

    string lineA, lineB, lineC;
    int rowCount = 0;

    while (getline(inFileA, lineA) &&
        getline(inFileB, lineB) &&
        getline(inFileC, lineC) &&
        rowCount < rowsToRead) {

        stringstream ssA(lineA);
        stringstream ssB(lineB);
        stringstream ssC(lineC);
        string hexStr;

        // Reset or initialize before each run
        resetarrays(h_b, h_a, N);
        c = 0;
        // Read 16 floats from A and B
         // Read 8 floats for h_a
        for (int i = 0; i < k; i++) {
            if (!(ssA >> hexStr)) break;
            uint32_t bits = stoul(hexStr, nullptr, 16);
            memcpy(&h_aa[i], &bits, sizeof(float));
            h_a[i] = (h_aa[i]);
        }

        // Read 8 floats for h_b
        for (int i = 0; i < k; i++) {
            if (!(ssB >> hexStr)) break;
            uint32_t bits = stoul(hexStr, nullptr, 16);
            memcpy(&h_bb[i], &bits, sizeof(float));
            h_b[i] = (h_bb[i]);
        }


        // Read 1 float from C
         // Read C (bit string)
        if (lineC.size() == 32) {
            uint32_t bits = bitset<32>(lineC).to_ulong();
            float h_cc = 0.0f;
            memcpy(&h_cc, &bits, sizeof(float));
            c = h_cc;  // store reconstructed float
        }
        // Run your CUDA operation
        cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

        mma_fp16_acc_fp32 << <1, 32 >> > (d_C, d_b, d_a, c);
        cudaDeviceSynchronize();
        cudaMemcpy(h_C, d_C, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

        // Write result for this row
        writeFloatBitsToFile(outfilename, h_C[0]);

        rowCount++;
    }


    inFileA.close();
    inFileB.close();
    inFileC.close();


    cudaFree(d_C);
    cudaFree(d_a);
    cudaFree(d_b);
    // cudaFree(d_c2);







}
