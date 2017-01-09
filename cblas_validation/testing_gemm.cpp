/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include <typeinfo>
#include <cblas.h>

using namespace std;

/* ============================================================================================ */

void read_txt_file(ifstream &inputFileA, vector<float> &hA, int lda)
{
     //input file  1 2 3 
     //            4 5 6
     //becuase it is column major, the output 1-D vector ={1, 4, 2, 5, 3, 6}
        string line;
        int row = 0;
        while ( getline (inputFileA, line) )
        {
            stringstream ss(line);
            string token;
            int col = 0;
            while (getline(ss, token, ' ')) //split the string to individual string
            {
                float value = (float)atoi(token.c_str());
                //cout << value << " ";
                hA[col * lda + row] = value;
                col++;
            }
            row++;
            //cout << endl;
        }
}



int main()
{

    int M, N, K;
    M = N = K = 192;

    int lda = M;
    int ldb = K;
    int ldc = M;

    int  A_size, B_size, C_size, A_row, A_col, B_row, B_col;
    float alpha = 1.0;
    float beta = 0.0;

    CBLAS_TRANSPOSE transA = CblasNoTrans;//CblasTrans
    CBLAS_TRANSPOSE transB = CblasTrans;

    if(transA == CblasNoTrans){
        A_row =  M; A_col = K;
    }
    else{
        A_row = K; A_col = M;
    }

    if(transB == CblasNoTrans){
        B_row =  K; B_col = N;
    }
    else{
        B_row = N; B_col = K;
    }

    A_size = lda * A_col; B_size = ldb * B_col; C_size = ldc * N;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<float> hA(A_size);
    vector<float> hB(B_size);
    vector<float> hC(C_size);

    /* read A and B from txt file, notice, A, B are col major */
    // C is not needed to be initialized, as beta = 0;

    ifstream inputFileA;
    inputFileA.open("A_col_major.txt");
    read_txt_file(inputFileA, hA, lda);
    inputFileA.close();   

    ifstream inputFileB;
    inputFileB.open("B_col_major.txt");
    read_txt_file(inputFileB, hB, ldb);
    inputFileB.close();  

    
    cblas_sgemm(     CblasColMajor,
                     transA, transB, M, N, K,
                     alpha, hA.data(), lda,
                     hB.data(), ldb,
                     beta, hC.data(), ldc); 

    ofstream outputFileC;

    outputFileC.open("result_C_col_major.txt");

    for(int row=0;row<M;row++){
        for(int col=0;col<N;col++){
            outputFileC << hC[row+col*ldc] << " ";
        }
        outputFileC << endl;
    }

    return 0;
}




