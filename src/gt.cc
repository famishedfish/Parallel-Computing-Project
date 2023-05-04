/* Copyright 2002 Ben Blum, Christian Shelton
 *
 * This file is part of GameTracer.
 *
 * GameTracer is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * GameTracer is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GameTracer; if not, write to the Free Software Foundation, 
 * Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "ipa.h"
#include "enum.h"
#include "gnm.h"
#include "nfgame.h"
#include "makegame.h"
#include <chrono>

// CONSTANTS
// For explanation of constants, refer to the appropriate header file

// GNM CONSTANTS
#define STEPS 100
#define FUZZ 1e-12
#define LNMFREQ 3
#define LNMMAX 10
#define LAMBDAMIN -10.0
#define WOBBLE 0
#define THRESHOLD 1e-2

// IPA CONSTANTS
#define ALPHA 0.02
#define EQERR 1e-6

void usage(char *name) { 
  cout << "GameTracer 0.2\n\
  usage: " << name << " [-i|-e] [-t] [-s size of nodes to degenerate to serial] [file|-r players actions gameseed] rayseed\n\
  \n\
  -i:      use IPA (iterative polymatrix approximation)\n\
  -e:      use enumerate (pure strategy)\n\
  -t:      use task to parallel\n\
  -s:      the size threshold  to degenrate from parallel to serial\n\
  -d:      set when use enumerate\n\
  file:    read game in from file\n\
  -r:      generate a game with the specified number of players and\n\
          actions per player, with payoffs chosen randomly from [0,1]\n\
  rayseed: random seed for the perturbation ray, g\n";
}

int main(int argc, char **argv) {
  auto start_time = std::chrono::high_resolution_clock::now();
  int i, seed, doipa = 0, doenum = 0, usetask = 0, argbase = 0, thresholdsize = 0;
  nfgame *A;
  // gnmgame *A;

  if(argc < 2) {
    usage(argv[0]);
    return -1;
  }
  if(strcmp(argv[1],"-i") == 0) {
    doipa = 1;
    argbase++;
    argc--;
    if(argc < 2) {
      usage(argv[0]);
      return -1;
    }
  }
  if(strcmp(argv[1],"-e") == 0) {
    doenum = 1;
    argbase++;
    argc--;
    if(argc < 2) {
      usage(argv[0]);
      return -1;
    }
  }
  if(strcmp(argv[1+argbase],"-t") == 0) {
    usetask = 1;
    argbase++;
    argc--;
    if(argc < 1) {
      usage(argv[0]);
      return -1;
    }
  }
  if(strcmp(argv[1+argbase],"-s") == 0) {
    thresholdsize = atoi(argv[2+argbase]);
    argbase += 2;
  }
  if(strcmp(argv[1+argbase],"-r") == 0) {
    if(argc < 6) {
      usage(argv[0]);
      return -1;
    }
    A = makeRandomNFGame(atoi(argv[2+argbase]),atoi(argv[3+argbase]),atoi(argv[4+argbase]));
    seed = atoi(argv[5+argbase]);
  } else {
    if(argc < 3) {
      usage(argv[0]);
      return -1;
    }
    A = makeNFGame(argv[1+argbase]);
    seed = atoi(argv[2+argbase]);
  }
  if(A == 0) {
    cout << "Unable to create game.\n";
    //    usage(argv[0]);
    return -1;
  }
  
  srand48(seed);
  cvector g(A->getNumActions()); // choose a random perturbation ray
  int numEq;
  if(doipa) {
    cout << "IPA\n";
    cvector ans(A->getNumActions());
    cvector zh(A->getNumActions(),1.0);
    do {
      for(i = 0; i < A->getNumActions(); i++) {
        g[i] = drand48();
      }
      g /= g.norm(); // normalized
      numEq = IPA(*A, g, zh, ALPHA, EQERR, ans);
    } while(numEq == 0);
    if(numEq)
      cout << ans << endl;
  } else if (doenum) {
    cout << "ENUM\n";
    // Find one NE
    start_time = std::chrono::high_resolution_clock::now();
    int ans[A->getNumPlayers()];
    if (ENUM(*A, ans, usetask, thresholdsize)) {
      for (int i = 0; i < A->getNumPlayers(); i++)
        cout << int(ans[i]) << " ";
      cout << "\n";
    }
    else
      cout << "No pure NE." << endl;
  } else {
    cout << "GNM\n";
    cvector **answers;
    do {
      for(i = 0; i < A->getNumActions(); i++) {
      g[i] = drand48();
      }
      g /= g.norm(); // normalized
      numEq = GNM(*A, g, answers, STEPS, FUZZ, LNMFREQ, LNMMAX, LAMBDAMIN, WOBBLE, THRESHOLD);
    } while(numEq == 0);
    for(i = 0; i < numEq; i++) {
      cout << *(answers[i]) << endl;
      delete answers[i];
    }
    free(answers);
  }
  delete A;

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  std::cout << "Time taken: " << duration << " ms" << std::endl;
}
