#include <algorithm>
#include "cmatrix.h"
#include "gnm.h"
#include "gnmgame.h"

int NENum = 0;

// Returns true if a given strategy profile is a Nash equilibrium
bool isNE(gnmgame &A, int* profile) {
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

void setProfileRecursive(gnmgame &A, int* profile, int player, int* ans) {
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
        // Loop over all possible strategies for the current player
        #pragma omp parallel for
        for (int i = 0; i < A.getNumActions(player); i++) {
            profile[player] = i;
            setProfileRecursive(A, profile, player+1, ans);
        }
    }
}

void setProfileIterative(gnmgame &A, int* ans) {
    int numPlayers = A.getNumPlayers();
    int profile[numPlayers];
    int player = 0;
    int i, j, k;
    while (true) {
        if (player == numPlayers) {
            if (isNE(A, profile)) {
                memcpy(ans, profile, sizeof(int) * numPlayers);
                NENum++;
            }
            // Increment the profile
            i = numPlayers - 1;
            while (i >= 0) {
                if (profile[i] < A.getNumActions(i) - 1) {
                    profile[i]++;
                    break;
                } else {
                    profile[i] = 0;
                    i--;
                }
            }
            if (i < 0) {
                // All profiles have been enumerated
                break;
            }
            player = i + 1;
        } else {
            // Set the profile for the current player
            profile[player] = 0;
            player++;
        }
    }
}

// Enumerates all possible strategy profiles and checks if they are Nash equilibria
// Return number of NEs
int ENUM(gnmgame &A, int* ans) {
    int numPlayers = A.getNumPlayers();
    int profile[numPlayers];
    // setProfileRecursive(A, profile, 0, ans);
    setProfileIterative(A, ans);
    return NENum;
}