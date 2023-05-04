#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "enummix2.cc"

struct GlobalConstants {
    cmatrix* A;
    cmatrix* B;
    int* q;
};

__constant__ GlobalConstants cuConstParams;
int* cuTheta;

vector<vector<int>> C;
void compute_combination(int n, int k) {
    C.resize(n+1, vector<int>(k+1, 0));
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= min(i, k); j++) {
            if (j == 0 || j == i) {
                C[i][j] = 1;
            } else {
                C[i][j] = C[i-1][j-1] + C[i-1][j];
            }
        }
    }
}

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

void setCuSupportSet(int* cuTheta, std::vector<std::vector<int>> theta, int q) {
    if (cuTheta != NULL) {
        cudaFree(cuTheta);
    }
    // Allocate memory on the device for the vector and its contents
    cudaMalloc(&cuTheta, C[q][1] * q * sizeof(int));   // C(q, 1) * q

    // Copy the contents of the vector from host memory to device memory
    size_t offset = 0;
    for (const auto& vec : theta) {
        size_t size = vec.size() * sizeof(int);
        cudaMemcpy(cuTheta + offset, vec.data(), size, cudaMemcpyHostToDevice);
        offset += vec.size();
    }
}

__device__ void argmax(double maxVal, double* p, int length, int* id, int count) {
    int j = 0;
    for (int i = 0; i < length; i++)
        if (p[i] == maxVal) id[j++] = i;
}

__global__ void pure(int* cuTheta, double* ans1, double* ans2) {
    int q = *cuConstParams.q;
    double *x, *y, *p1, *p2;
    cudaMalloc(&x, sizeof(double) * q);
    cudaMalloc(&y, sizeof(double) * q);
    cudaMalloc(&p1, sizeof(double) * q);
    cudaMalloc(&p2, sizeof(double) * q);
    for (int i = 0; i < q; i++) {
        x[i] = y[i] = p1[i] = p2[i];
    }
    for (int i = 0; i < q; i++)
        x[i] = cuTheta[q * blockIdx.x + i];
    double maxP1, count1 = 0;
    (*((uint64_t*)&maxP1))= ~(1LL<<52);
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            p1[i] += x[j] * cuConstParams.B->x[j*q+i];
        }
        maxP1 = p1[i] > maxP1 ? p1[i] : maxP1;
        count1 = p1[i] > maxP1 ? 1 : (p1[i] == maxP1 ? count1 + 1 : count1);
    }
    int* idx;
    cudaMalloc(&idx, sizeof(int) * count1);
    argmax(maxP1, p1, q, idx, count1);

    double maxP2, count2 = 0;
    (*((uint64_t*)&maxP2))= ~(1LL<<52);
    for (int i = 0; i < q; i++)
        y[i] = cuTheta[q * threadIdx.x + i];
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            p2[i] += y[j] * cuConstParams.A->x[i*q+j];
        }
        maxP2 = p2[i] > maxP2 ? p2[i] : maxP2;
        count2 = p2[i] > maxP2 ? 1 : (p2[i] == maxP2 ? count2+ 1 : count2);
    }
    int* idy;
    cudaMalloc(&idy, sizeof(int) * count2);
    argmax(maxP2, p2, q, idy, count2);
    
    for (int i = 0; i < count1; i++) {
        for (int j = 0; j < count2; j++) {
            if (cuTheta[q * blockIdx.x + idy[j]] && cuTheta[q * threadIdx.x + idx[i]]) {
                // This is an NE
                memcpy(ans1, &x, sizeof(double) * q);
                memcpy(ans2, &y, sizeof(double) * q);
            }
        }
    }
}

int ENUMMIX(gnmgame &G, std::vector<std::vector<cvector> > &ans) {
    cmatrix A = G.getPurePayoffMatrix(0);
    cmatrix B = G.getPurePayoffMatrix(1);


    unsigned int p1NumActions = G.getNumActions(0);
    unsigned int p2NumActions = G.getNumActions(1);
    unsigned int q = p1NumActions < p2NumActions ? p1NumActions : p2NumActions;

    compute_combination(q, q);

    setup(A, B, q);

    std::vector<std::vector<int>> theta;
    getKSupportSet(q, 1, theta);
    setCuSupportSet(cuTheta, theta, q);
    theta.clear();

    cout << theta[0][0] << " " << theta[0][1] << " " << theta[1][0] << " " << theta[1][1];
    double *ans1, *ans2;
    cudaMalloc(&ans1, sizeof(double) * q);
    cudaMalloc(&ans2, sizeof(double) * q);
    dim3 blockDim(C[q][1], 1, 1);
    dim3 gridDim(C[q][1], 1, 1);
    pure<<<gridDim, blockDim>>>(cuTheta, ans1, ans2);
    // for (int k = 2; k <= q; k++) {
    //     getKSupportSet(q, k, theta);
    //     // set up theta
    //     // run<<>>(k)
    //     cudaDeviceSynchronize();
    //     // ans = ans union (A, B, k, q, theta)
    // }

    // return 0;
}