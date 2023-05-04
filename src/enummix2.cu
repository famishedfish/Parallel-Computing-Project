#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "enummix2.h"

struct GlobalConstants {
    cmatrix* A;
    cmatrix* B;
    int* q;
};

__constant__ GlobalConstants cuConstParams;
__constant__ int* theta;

void setup(cmatrix A, cmatrix B, int q) {
    GlobalConstants params;
    double *dataA, *dataB;

    cudaMalloc(&dataA, sizeof(double) * A.getSize());
    cudaMalloc(&dataB, sizeof(double) * A.getSize());
    cudaMalloc(&params.A, sizeof(A));
    cudaMalloc(&params.B, sizeof(B));
    cudaMalloc(&params.q, sizeof(int));

    cudaMemcpy(params.A, &A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(params.B, &B, sizeof(B), cudaMemcpyHostToDevice);
    cudaMemcpy(params.q, &q, sizeof(int), cudaMemcpyHostToDevice);

    params.A->setx(dataA);
    params.A->setx(dataB);

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));
}

int ENUMMIX(gnmgame &G, std::vector<std::vector<cvector> > &ans) {
    int n = G.getNumActions();
    cmatrix A = G.getPurePayoffMatrix(0);
    cmatrix B = G.getPurePayoffMatrix(1);

    unsigned int p1NumActions = G.getNumActions(0);
    unsigned int p2NumActions = G.getNumActions(1);
    unsigned int q = p1NumActions < p2NumActions ? p1NumActions : p2NumActions;

    setup(A, B, q);

    // theta = Generate(k, q)
    // theta = getKSupportSet(n, );
    // ans = Pure(A, B, q, theta)
    for (int k = 2; k <= q; k++) {
        // theta = Generate(k, q)
        // set up theta
        // run<<>>(k)
        // ans = ans union (A, B, k, q, theta)
    }

    // return 0;
}