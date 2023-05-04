#include "cmatrix.h"
#include "enummix2.h"
#include "gnmgame.h"

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


// recursive helper function to find KSupportSet
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


// Compute NE by solving Mx = b
bool solveNE(cmatrix &A, std::vector<int> &Mx, std::vector<int> &My, cvector &y) {
    // M_(m,n) = 3 types of constriants: indifference, zero support, prob sum up to 1
    int m = Mx.size() - 1 + A.getn() - My.size() + 1;
    int n = A.getn();
    double *coef = new double[m * n]();
    // indifference constraints
    cvector v0(A[Mx[0]], n);
    for (int i = 1; i < Mx.size(); ++i) {
        cvector vi(A[Mx[i]], n);
        cvector vdiff = v0 - vi;
        memcpy(coef + (i - 1) * n, vdiff.values(), n * sizeof(double));
    }
    // zero constraints
    vector<int> zero(n, 1);
    vector<double> zeroCoef(n, 0);
    for (int i = 0; i < My.size(); ++i) {
        zero[My[i]] = 0;
    }
    int ridx = Mx.size() - 1;
    for (int i = 0; i < n; ++i) {
        if (zero[i] == 1) {
            zeroCoef[i] = 1;
            memcpy(coef + ridx * n, zeroCoef.data(), n * sizeof(double));
            ridx++;
            zeroCoef[i] = 0;
        }
    }
    // sum(prob) == 1 constraint
    vector<double> probCoef(n, 1);
    memcpy(coef + (m - 1) * n, probCoef.data(), n * sizeof(double));

    cmatrix M(coef, m, n);

    // b_(m,1) = [0,0,0...0,1]
    double *res = new double[m]();
    res[m - 1] = 1;
    
    cvector b(res, m);

    // Solve the equations
    if (M.solve(b, y)) {
        for (int t : My) {
            if (y[t] <= 0) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// Verify NE with threorem 1
bool isNE(cvector &x, cvector &y, vector<int> Mx, vector<int> Ny, cmatrix A, cmatrix B) {
    cvector u = A * y;
    cvector v = cmatrix(B, true) * x;
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

int ENUMMIX2(gnmgame &G, std::vector<std::vector<cvector> > &ans) {
    cmatrix A = cmatrix(G.getPurePayoffMatrix(0), true);
    cmatrix B = cmatrix(G.getPurePayoffMatrix(1), true);

    unsigned int p1NumActions = G.getNumActions(0);
    unsigned int p2NumActions = G.getNumActions(1);
    unsigned int k = min(p1NumActions, p2NumActions);

    for (int i = 1; i <= k; ++i) {
        std::vector<std::vector<int>> p1KsupportSet, p2KsupportSet;
        getKSupportSet(p1NumActions, i, p1KsupportSet);
        getKSupportSet(p2NumActions, i, p2KsupportSet);

        for (std::vector<int> Mx : p1KsupportSet) {
            for (std::vector<int> My : p2KsupportSet) {
                cvector x(p1NumActions), y(p2NumActions);
                bool s1 = solveNE(A, Mx, My, y);
                cmatrix BT = cmatrix(B, true);
                bool s2 = solveNE(BT, My, Mx, x);
                if (s1 && s2 && isNE(x, y, Mx, My, A, B)) {
                    ans.push_back({x, y});
                }
            }
        }
    }
    return ans.size();
}