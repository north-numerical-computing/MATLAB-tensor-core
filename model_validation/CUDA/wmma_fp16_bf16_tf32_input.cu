/* Test numerical features of CUDA wmma instruction.

   This test bench is based on the code in
   https://github.com/north-numerical-computing/tensor-cores-numerical-behavior

   Reference:
     Faizan A Khattak and Mantas Mikaitis,
     Generalized Methodology for Determining Numerical Features of Hardware
     Floating-Point Matrix Multipliers: Part I.
     Accepted for 29th Annual IEEE High Performance Extreme Computing. Sep. 2025.
 */

#include <assert.h>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <bitset>
#include <mma.h>
#include <iomanip>
#include <cuda_bf16.h>
#include <cuda_fp16.h>


#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;
using namespace nvcuda;

/*------------ Parameter Selection by User---------------------------------*/
// 1. Choose intype format and uncomment it
#define IN_FORMAT_BF16
//#define IN_FORMAT_FP16
//#define IN_FORMAT_TF32


//2. Chosse output format, fp16 only for fp16 input, 
//#define OU_FORMAT_FP16
#define OUT_FORMAT_FP32


//3. Choose GPU model and uncomment it that you are targeting
//#define B200
//#define A100
//#define A2
//#define A30
//#define A40
#define L40S
//#define H200
//#define H100
//#define Ada
//#define V100

/*------------ End of Parameter Selection by User---------------------------------*/





//#############################################
/* Set wmma shape */
#define M 16
#define N 16
int pout = 24;

/* Base on the input format, set up various parameters. */
#ifdef IN_FORMAT_BF16
#define IN_FORMAT nv_bfloat16
#define WMMA_IN_FORMAT nv_bfloat16
#define CONVERSION_OP  __float2bfloat16
#define CONVERSION_RE __bfloat162float
int pin = 8;
#define K 16
#elif defined(IN_FORMAT_FP16)
#define IN_FORMAT half
#define WMMA_IN_FORMAT half
#define CONVERSION_OP  __float2half
#define CONVERSION_RE __half2float
int pin = 11;
int pout16 = 11;
#define K 16
#elif defined(IN_FORMAT_TF32)
#define IN_FORMAT float
#define WMMA_IN_FORMAT wmma::precision::tf32
int pin = 11;
#define CONVERSION_OP
#define CONVERSION_RE
#define K 8
#endif






/* Set up the output format */
#define OUT_FORMAT float


/****************************************************
 * Memory management and wmma::mma_sync() interface *
 ****************************************************/

 /* Set the entries of host arrays to zero. */
template <typename returntype>
void host_reset(IN_FORMAT* a, IN_FORMAT* b, returntype* c) {
    memset(a, 0, 16 * 16 * sizeof(IN_FORMAT));
    memset(b, 0, 16 * 16 * sizeof(IN_FORMAT));
    memset(c, 0, 16 * 16 * sizeof(returntype));
}


/* Compute C += A*B, where A, B, and C are MxN matrices.
   The matrix C is initialized to 0 when `init` is true. */
template <typename returntype>
__global__ void wmma_ker(IN_FORMAT* a, IN_FORMAT* b,
    returntype* c, bool init) {

    // Declare fragments.
    wmma::fragment<wmma::matrix_a, M, N, K, WMMA_IN_FORMAT,
        wmma::row_major> a_fragment;
    wmma::fragment<wmma::matrix_b, M, N, K, WMMA_IN_FORMAT,
        wmma::col_major> b_fragment;
    wmma::fragment<wmma::accumulator, M, N, K, returntype> c_fragment;

    // Load input matrices and initialize output (if required).
    wmma::load_matrix_sync(a_fragment, a, N);
    wmma::load_matrix_sync(b_fragment, b, M);
    if (init)
        wmma::fill_fragment(c_fragment, 0.0f);
    else
        wmma::load_matrix_sync(c_fragment, c, N, wmma::mem_col_major);

    // Multiply
    wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

    // Store the output
    wmma::store_matrix_sync(c, c_fragment, N, wmma::mem_col_major);
}


/* Copy data from host to device, perform the operation, and copy result back to
   host. */
template <typename returntype>
void wmma_init_run(IN_FORMAT* h_a, IN_FORMAT* h_b, returntype* h_c,
    IN_FORMAT* d_a, IN_FORMAT* d_b, returntype* d_c,
    bool init) {

    // Copy input from host to device.
    cudaMemcpy(d_a, h_a, 16 * 16 * sizeof(IN_FORMAT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 16 * 16 * sizeof(IN_FORMAT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, 16 * 16 * sizeof(returntype), cudaMemcpyHostToDevice);

    // Perform matrix multiplication.
    wmma_ker << <1, 32 >> > (d_a, d_b, d_c, init);

    // Copy result from device to host.
    cudaMemcpy(h_c, d_c, 16 * 16 * sizeof(returntype), cudaMemcpyDeviceToHost);
}


/**********************
 * Printing functions *
 **********************/
void printheader(FILE* outfile, const char* string) {
    fprintf(outfile,
        "+--------------------------------------------------------------+\n");
    fprintf(outfile, "| %-60s |\n", string);
    fprintf(outfile,
        "+--------------------------------------------------------------+\n");
}
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


void writeFloatToFile(const string& filename, float value) {
    ofstream outFile(filename, ios::app);  // open in append mode
    if (!outFile) {
        cerr << "Error opening file!" << endl;
        return;
    }

    outFile << setprecision(7) << value << endl;  // full float precision
    outFile.close();
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







/***************
 * EXPERIMENTS *
 ***************/
int main(int argc, char** argv) {

    IN_FORMAT* h_a, * h_b, * h16_c, * d16_a, * d16_b, * d16_c;
    OUT_FORMAT* d_c, * h_c;
    h_a = new IN_FORMAT[16 * 16];
    h_b = new IN_FORMAT[16 * 16];
    h_c = new OUT_FORMAT[16 * 16];
    h16_c = new IN_FORMAT[M * N];

    cudaMalloc(&d16_a, 16 * 16 * sizeof(IN_FORMAT));
    cudaMalloc(&d16_b, 16 * 16 * sizeof(IN_FORMAT));
    cudaMalloc(&d16_c, M * N * sizeof(IN_FORMAT));
    cudaMalloc(&d_c, 16 * 16 * sizeof(OUT_FORMAT));

    FILE* outfile = stdout;

    
    string intype = "fp16";//default
    string GPUmodel = "H200"; // default
    string outtyped = "fp16"; //
    string outtypec = "fp32";

// intype macros
#ifdef IN_FORMAT_FP16
    intype = "fp16";
    outtypec = "fp32"; // assuming c can be rounded to fp16 by CUDA using float2half as rne
#ifdef OUT_FORMAT_FP16
    outtyped = "fp16";
#endif
#ifdef OUT_FORMAT_FP32
    outtyped = "fp32";
#endif
#endif // 
#ifdef IN_FORMAT_BF16
    intype = "bf16";
    outtypec = "fp32";
    outtyped = "fp32";
#endif // 
#ifdef IN_FORMAT_TF32
    intype = "tf32";
    outtypec = "fp32";
    outtyped = "fp32";
#endif // 




    // Models pick from macros
#if defined(B200)
    GPUmodel = "B200";
#elif defined(A100)
    GPUmodel = "A100";
#elif defined(A2)
    GPUmodel = "A2";
#elif defined(A30)
    GPUmodel = "A30";
#elif defined(A40)
    GPUmodel = "A40";
#elif defined(L40S)
    GPUmodel = "L40S";
#elif defined(H200)
    GPUmodel = "H200";
#elif defined(H100)
    GPUmodel = "H100";
#elif defined(Ada)
    GPUmodel = "Ada";
#elif defined(V100)
    GPUmodel = "V100";
#else
    GPUmodel = "Unknown";
#endif









    string path;
    ///////////////////////////////////////////////////////////////////////////////
    
        path = ""; // uncomment for GPU results
        path = path + GPUmodel + "/" + intype + "/";


    

    string outfilename = path + "d_" + GPUmodel + "_" + outtyped + ".txt";


    int fma_size = 4;  // default
    if (GPUmodel == "V100")
    {
        fma_size = 4;
    }
    // A100-A2-L40S-Ada
    if (GPUmodel == "A100" | GPUmodel == "A2" | GPUmodel == "L40S" | GPUmodel == "Ada")
    {
        if (intype == "fp16") { fma_size = 8; }
        if (intype == "bf16") { fma_size = 8; }
        if (intype == "tf32") { fma_size = 4; }
    }

    if (GPUmodel == "H100" | GPUmodel == "H200" | GPUmodel == "B200")
    {
        if (intype == "fp16") { fma_size = 16; }
        if (intype == "bf16") { fma_size = 16; }
        if (intype == "tf32") { fma_size = 4; }
    }

    
    ifstream inFileA(path + "a_" + GPUmodel + "_" + intype + ".txt"); // a
    ifstream inFileB(path + "b_" + GPUmodel + "_" + intype + ".txt"); // b

    ifstream inFileC(path + "c_" + GPUmodel + "_" + outtypec + ".txt");

    if (!inFileA || !inFileB || !inFileC) {
        cerr << "Error opening one or more files!" << endl;
        return 1;
    }
    float h_aa[16], h_bb[16];
    float h_cc[1];     // single element from c file
    int rowsToRead = 100000;

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
        host_reset(h_a, h_b, h_c);
        host_reset(h_a, h_b, h16_c);

        // Read 16 floats from A and B
         // Read 8 floats for h_a
        for (int i = 0; i < fma_size; i++) {
            if (!(ssA >> hexStr)) break;
            uint32_t bits = stoul(hexStr, nullptr, 16);
            memcpy(&h_aa[i], &bits, sizeof(float));
            h_a[i] = CONVERSION_OP(h_aa[i]);
        }

        // Read 8 floats for h_b
        for (int i = 0; i < fma_size; i++) {
            if (!(ssB >> hexStr)) break;
            uint32_t bits = stoul(hexStr, nullptr, 16);
            memcpy(&h_bb[i], &bits, sizeof(float));
            h_b[i] = CONVERSION_OP(h_bb[i]);


        }

        //for debugging




        // Read 1 float from C
         // Read C (bit string)
        if (lineC.size() == 32) {
            uint32_t bits = bitset<32>(lineC).to_ulong();
            float h_cc = 0.0f;
            memcpy(&h_cc, &bits, sizeof(float));
            if (outtypec == outtyped) {
                h_c[0] = h_cc;  // store reconstructed float
            }
            else {
                h16_c[0] = CONVERSION_OP (h_cc);


            }

        }
        // Run your CUDA operation
        if (outtypec == outtyped)
        {
            wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        }

        {
          #ifdef IN_FORMAT_FP16
                    wmma_init_run(h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
            #endif // DEBUG

        }

        // Write result for this row
        if (outtypec == outtyped)
        {
            writeFloatBitsToFile(outfilename, h_c[0]);
        }
        else
        {
        #ifdef IN_FORMAT_FP16
             writeFloatBitsToFile(outfilename, __half2float(h16_c[0]));
        #endif      
        }
        rowCount++;

    }

    

    inFileA.close();
    inFileB.close();
    inFileC.close();



    /* Free dynamically allocated memory. */
    free(h_a);
    free(h_b);
    free(h16_c);
    cudaFree(d16_a);
    cudaFree(d16_b);
    cudaFree(d16_c);
    cudaFree(d_c);
    free(h_c);
}

