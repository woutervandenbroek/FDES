/*==================================================================

Copyright (C) 2015 Wouter Van den Broek, Xiaoming Jiang

This file is part of FDES.

FDES is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FDES is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FDES. If not, see <http://www.gnu.org/licenses/>.

Email: wouter.vandenbroek@uni-ulm.de, wouter.vandenbroek1@gmail.com,
       xiaoming.jiang@uni-ulm.de, jiang.xiaoming1984@gmail.com 

===================================================================*/

#ifndef optimFunctions_lkjhdqzutew
#define optimFunctions_lkjhdqzutew

#include <cuComplex.h>
#include <cufft.h>
#include "paramStructure.h"
#include "kernel_assert.hpp"


__global__ void copyMiddleOut(float* Imodel, cuComplex* psi, params_t* params);

__global__ void copyMiddleIn(cuComplex* dE, float* Imodel, params_t* params);


void progressCounter(int j, int jTot);

void initialPotential(cufftComplex* V_d, params_t* params);

bool isPowOfTwo( int j );

bool isPowOfHalf( int j );

#endif