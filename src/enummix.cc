#include "cmatrix.h"
#include "enummix.h"
#include "gnmgame.h"
#include<vector>

// Input: n - Number of action
// Output: powerset - all supports sets
void getPowerSet(int n, std::vector<std::vector<int> > &powerset) {
    unsigned int setsize = pow(2, n);
    powerset.resize(setsize);
    for (int i = 0; i < setsize; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i & (1 << j)) {
                powerset[i].push_back(j);
            }
        }
    }
}


void kComb(int n, int k, int i, int x, std::vector<std::vector<int>> &output, std::vector<int> tmp) {
    if (i == k) {
        output.push_back(tmp);
        return;
    }
    if (x >= n) {
        return;
    }
    kComb(n, k, i, x + 1, output, tmp);
    tmp[i] = x;
    kComb(n, k, i + 1, x + 1, output, tmp);
}

// Input: n - Number of action, k - Support size
// Output: ksupportset - suppport sets of size K
void getKSupportSet(int n, int k, std::vector<std::vector<int>> &ksupportset) {
    std::vector<int> tmp(k, 0);
    kComb(n, k, 0, 0, ksupportset, tmp);
}

bool solveNE(cmatrix &A, cvector &Mx, cvector &My, cvector &y) {
    int m = Mx.getm() - 1 + A.getn() - My.getm() + 1;
    int n = A.getn();

    double *coef = new double[m * n];

    // indifference constraints
    cvector vold(A[Mx[0]], n);
    for (int i = 1; i < Mx.getm(); ++i) {
        cvector vnew(A[Mx[i]], n);
        cvector vdiff = vold - vnew;
        memcpy(coef + (i - 1) * n, vdiff.values(), n * sizeof(double));
    }

    // zero constraints
    vector<int> zero(n, 1);
    vector<double> zeroCoef(n, 0);
    for (int i = 0; i < My.getm(); ++i) {
        zero[int(My[i])] = 0;
    }
    for (int i = 0; i < n; ++i) {
        if (zero[i] == 1) {
            zeroCoef[i] = 1;
            memcpy(coef + (Mx.getm() + i) * n, zeroCoef.data(), n * sizeof(double));
            zeroCoef[i] = 0;
        }
    }

    // sum(prob) == 1 constraint
    vector<double> probCoef(n, 1);
    memcpy(coef + (m - 1) * n, probCoef.data(), n * sizeof(double));

    cmatrix M(coef, m, n);

    // 000...01
    double *res = new double[m]();
    res[m - 1] = 1;
    cvector b(res, m);

    // Solve the equations
    if (M.solve(b, y) && y.min() > 0) {
        return true;
    } else {
        return false;
    }
}


// Verify NE with threorem 1
bool isNE(cvector &x, cvector &y, vector<int> Mx, vector<int> Ny, cmatrix A, cmatrix B) {
    cvector u = A * y;  // shape = (actions[0])
    cvector v = cmatrix(B, true) * x;  // shape = (actions[1])
    double umax = u.max();
    for (int action : Mx) {
        if (u[action] != umax) {
            return false;
        }
    }
    double vmax = v.max();
    for (int action : Ny) {
        if (v[action] != vmax) {
            return false;
        }
    }
    return true;
}

int ENUMMIX(gnmgame &G, int* ans) {
    cmatrix A = G.getPurePayoffMatrix(0);
    cmatrix B = G.getPurePayoffMatrix(1);

    unsigned int p1NumActions = G.getNumActions(0);
    unsigned int p2NumActions = G.getNumActions(1);
    unsigned int k = min(p1NumActions, p2NumActions);

    for (int i = 1; i <= k; ++i) {
        std::vector<std::vector<int>> p1KsupportSet, p2KsupportSet;
        getKSupportSet(p1NumActions, i, p1KsupportSet);
        getKSupportSet(p2NumActions, i, p2KsupportSet);
        for (std::vector<int> Mx : p1KsupportSet) {
            for (std::vector<int> My : p2KsupportSet) {
                bool s1 = solveNE
                bool s2 = solveNE
            }
        }
    }


    for (std::vector<int> Mx : M)
        for (std::vector<int> Ny : N) {
            if (M.size() == 0 || N.size() == 0 || Mx.size() != Ny.size()) {
                continue;
            }

            cvector x(p1NumActions), y(p2NumActions);
            bool s1 = solveNE(cmatrix(B, true), cvector(Mx.data(), Mx.size()), cvector(Mx.data(), Mx.size()), y);
            bool s2 = solveNE(A, cvector(Mx.data(), Mx.size()), cvector(My.data(), My.size()), x);

            if (s1 && s2 && isNE(x, y, Mx, My, A, B)) {
                ans.
            }
        }
}