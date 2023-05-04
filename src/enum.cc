#include <algorithm>
#include "cmatrix.h"
#include "gnm.h"
#include "nfgame.h"
// #include "gnmgame.h"

int NENum = 0;
bool useTask = false;
int sizeThred = 1000;

// Returns true if a given strategy profile is a Nash equilibrium
bool isNE(nfgame &A, int* profile) {
    int numPlayers = A.getNumPlayers();
    for (int i = 0; i < numPlayers; i++) {
        double myPayoff = A.getPurePayoff(i, profile);
        int old_action = profile[i];
        // cout << "payoff is " << myPayoff << endl;
        for (int j = 0; j < A.getNumActions(i); j++) {
            profile[i] = j;
            double otherPayoff = A.getPurePayoff(i, profile);
            // cout << "[" << profile[0] << profile[1] << "] " << "player " << i << " other Payoff: " << otherPayoff << endl;
            if (myPayoff < otherPayoff) {
                profile[i] = old_action;
                return false;
            }
        }
        profile[i] = old_action;
    }
    // cout << "found\n";
    return true;
}

void setProfileRecursiveDegenerate(nfgame &A, int* profile, int player, int* ans) {
    int numPlayers = A.getNumPlayers();
    if (player == numPlayers) {
        // The profile has been set, check if it is an NE
        // cout << "print profile." << endl;
        // for (int i = 0; i < numPlayers; i++)
        //     cout << profile[i] << " ";
        // cout << endl;
        if (isNE(A, profile)) {
            memcpy(ans, profile, sizeof(int) * numPlayers);
            NENum++;
        }
    } else {
        int nodeSize = A.getBlockSize(player);
        #pragma omp parallel for
        for (int i = 0; i < A.getNumActions(player); i++) {
            profile[player] = i;
            if (nodeSize < sizeThred) {
                #pragma omp task
                setProfileRecursiveDegenerate(A, profile, player+1, ans);
            } else {
                setProfileRecursiveDegenerate(A, profile, player+1, ans);
            }
        }
    }
}

void setProfileRecursiveForParallel(nfgame &A, int* profile, int player, int* ans) {
    int numPlayers = A.getNumPlayers();
    if (player == numPlayers) {
        if (isNE(A, profile)) {
            memcpy(ans, profile, sizeof(int) * numPlayers);
            NENum++;
        }
    } else {
        // Loop over all possible strategies for the current player
        #pragma omp parallel for
        for (int i = 0; i < A.getNumActions(player); i++) {
            profile[player] = i;
            setProfileRecursiveForParallel(A, profile, player+1, ans);
        }
    }
}

// Enumerates all possible strategy profiles and checks if they are Nash equilibria
// Return number of NEs
int ENUM(nfgame &A, int* ans, bool taskFlag, int threshold) {
    int numPlayers = A.getNumPlayers();
    int profile[numPlayers];
    useTask = taskFlag;
    if (threshold)
        sizeThred = threshold;
    #ifdef _OPENMP
    cout << "Openmp available." << endl;
    #else
    cout << "Openmp not available." << endl;
    #endif
    // #pragma omp parallel
    // #pragma omp single
    // setProfileRecursiveDegenerate(A, profile, 0, ans);
    // return 0;
    int method = 3;
    switch (method) {
    case 1:
        // ---------- 1. pure parallel ------------------------
        setProfileRecursiveForParallel(A, profile, 0, ans);
        break;
    case 2:
        // ---------- 2. task degenerate ----------------------
        #pragma omp parallel
        #pragma omp single
        setProfileRecursiveDegenerate(A, profile, 0, ans);
        break;
    case 3:
        // ---------- 3. task * 1 + parallel ------------------
        #pragma omp parallel
        #pragma omp task
        for (int i = 0; i < A.getNumActions(0); i++) {
            profile[0] = i;
            setProfileRecursiveForParallel(A, profile, 1, ans);
        }
        break;
    case 4:
        // ---------- 4. parallel + task + parallel -----------
        break;
    }

    return NENum;
}