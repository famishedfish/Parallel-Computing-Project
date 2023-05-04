#ifndef __ENUMMIX2_H
#define __ENUMMIX2_H

#include <vector>
#include "cmatrix.h"
#include "gnmgame.h"

int ENUMMIX2(gnmgame &A, std::vector<std::vector<cvector> > &ans); 
void getKSupportSet(int n, int k, std::vector<std::vector<int>> &ksupportset);

#endif