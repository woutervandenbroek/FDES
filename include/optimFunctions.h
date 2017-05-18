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

__global__ void trialPotential (cuComplex* V, params_t* params);

__global__ void addConst(cuComplex* dst, cuComplex cst, int size);

__global__ void flipPotentialDevice (cuComplex* V, cuComplex thold, int size);

__global__ void flipPotentialAbsVals_d (cuComplex* V, cuComplex thold, int size);

__global__ void applyThreshold_d ( cuComplex* V, cuComplex thold, int size );

__global__ void copyMiddleOut(float* Imodel, cuComplex* psi, params_t* params);

__global__ void copyMiddleIn(cuComplex* dE, float* Imodel, params_t* params);

__global__ void copyDefoci_d ( params_t* params_d, int k, float defocus_k );

__global__ void copyTiltspec_d ( params_t* params, int k, float t0, float t1 );

__device__ float absoluteValueOF( float x );

__global__ void limitDerivatives_d ( cufftComplex* dEdV, cufftComplex* V, int size, float thrLo, float thrHi );

__global__ void signValues_d ( float* f, int size );

__global__ void ringFilter_d ( cufftComplex* f, int dim1, int dim2 );

__global__ void copySingle2Double ( double* fD, float* fS,  int size );

__global__ void divideBySqrt_d ( float* f0, float* f1, int size );


void saveResults(cufftComplex* V, float* E, float* L1, int j, params_t* params, int n3);

void randShuffle(int* t, int n);

bool smallErrorDiff( float* E_h, int j );

void initialUniformPDD( cufftComplex* f_d, float a, float b, int size );

void initialSparsePDD( cufftComplex* f_d, float b, params_t* params );

void flippedPotential(cuComplex* V, int it, params_t* params, params_t* params_d );

void flippedPotentialAbsVals ( cuComplex* V, params_t* params, params_t* params_d );

void negAbsRampup( params_t* params, params_t* params_d, int j );

float relaxParam( int it, float itH );

void setRelaxParam( params_t* params, params_t* params_d, int j );

cuComplex thresHoldInternal ( cuComplex* V, params_t* params, params_t* params_d, int flag );

cuComplex thresHold ( cuComplex* V, params_t* params, params_t* params_d );

cuComplex thresHoldAbsVals ( cuComplex* V, params_t* params, params_t* params_d );

int progressCounter(int j, params_t* params);

void readInitialPotential(cufftComplex* V_d, params_t* params);

void readIncomingWave ( cufftComplex* V_d, int k, params_t* params );

bool isPowOfTwo( int j );

bool isPowOfHalf( int j );

#endif