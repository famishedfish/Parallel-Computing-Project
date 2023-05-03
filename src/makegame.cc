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

#include "nfgame.h"
#include<fstream>

nfgame *makeRandomNFGame(int n, int actions, int seed) {
  int sizes[n];
  srand48(seed);
  int total = n;
  for(int i = 0; i < n; i++) {
    sizes[i] = actions;
    total *= actions;
  }
  cvector payoffs(total);
  for(int i = 0; i < total; i++) {
    payoffs[i] = drand48();
  }
  return new nfgame(n,sizes,payoffs);
}

nfgame *makeNFGame(char *filename) {
  int i, n, total, actions;
  ifstream in(filename);

  /* Now read in a game file.  The format for a game file is:
     numPlayers
     numActions[0] numActions[1] ... numActions[numPlayers-1]
     payoff(0,0,...,0) payoff(1,0,...,0) ... payoff(numActions[0]-1,numActions[1]-1,...,numActions[numPlayers-1]-1)
  */
  if(in.good() && !in.eof()) {
    in >> n;
    int size[n];
    total = n;
    actions = 0;
    for(i = 0; i < n; i++) {
      if(in.eof()) {
	cout << "Error in game file: not enough actions.\n";
	return 0;
      }
      in >> size[i];
      total *= size[i];
      actions += size[i];
    }
    if(total <= 0) {
      cout << "Error in game file: illegal actions.\n";
      return 0;
    }
      
    cvector payoffs(total);
    for(i = 0; i < total; i++) {
      if(in.eof()) {
	cout << "Error in game file: not enough payoffs.\n";
	return 0;
      }
      in >> payoffs[i];
    }	
    return new nfgame(n, size, payoffs);
  } else {
    cout << "Bad game file.\n";
    return 0;
  }
}
  
