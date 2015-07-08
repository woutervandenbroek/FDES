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

#ifndef projectedPotential_kzgfhbgw564gdlg74sg43jf96mhhse3ns
#define projectedPotential_kzgfhbgw564gdlg74sg43jf96mhhse3ns

#include <cufft.h>
#include "paramStructure.h"
#include "cufft_assert.h"
#include "cublas_assert.hpp"
#include "cuda_assert.hpp"
#include "coordArithmetic.h"
#include "complexMath.h"
#include "float.h"


__global__ void projectedPotential_d ( cufftComplex* V, int Z, params_t* params_d );

__device__ void parametersKirkland_d ( float* a, float* b, float* c, float* d, int Z, int i );

__global__ void normalizeProjectedPotential_d ( cufftComplex* V, float V0x, int size );


void normalizeProjectedPotential ( cufftComplex* V, int size, int gS, int bS );

//void writeCode ();

//void writeMATLABCode();

//void writeDeviceCode();

#endif