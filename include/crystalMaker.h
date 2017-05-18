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

#ifndef crystalMaker_liugjnblvdfliuwernlkhafdliuzgqwlkjhbgasouh
#define crystalMaker_liugjnblvdfliuwernlkhafdliuzgqwlkjhbgasouh

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <float.h>
#include "float.h"
#include <math.h>
#include <string.h>
#include <curand.h>
#include "cuda_assert.hpp"
#include "cublas_assert.hpp"
#include "paramStructure.h"
#include "curand_assert.h"
#include "curand_kernel.h"
#include "complexMath.h"
#include "projectedPotential.h"
#include "rwBinary.h"
#include "multisliceSimulation.h"
#include "optimFunctions.h"
#include "performanceTimer.h"

__global__ void setupCurandState_d(curandState *state, int seed, int size);

__global__ void atomJitter_d( float* xyz_d, float* dwf_d, int nAt, curandState *state );

__global__ void ascombeNoise_d( float* f, float dose, int size, curandState *state );

__global__ void squareAtoms_d ( cufftComplex* V, params_t* params, int nAt, int* Z, int Z0, float* xyz, float imPot, float* occ, int zlayer );

__device__ int mySignum_d( float x );

__global__ void divideBySinc ( cufftComplex* V2_d, params_t* params );

__global__ void multiplyWithProjectedPotential_d ( cufftComplex* V1, cufftComplex* V2, params_t* params );

__global__ void shiftCoordinates_d ( float* xyzCoord_d, float sX, float sY, int size );

__global__ void multiplyDetMask_d( float* J_d, params_t* params_d );


void buildMeasurements( params_t* params, int* Z_d, float* xyzCoord_d, float* DWF_d, float* occ_d );

void tiltCoordinates( float* xyz_d, int nAt, float t_0, float t_1, float t_2, params_t* params );

void atomJitter( float* xyz_d, curandState* state, int nAt, float* DWF_d );

void myGBSize( int* gbS, int size );

int myGSize( int size );

int myBSize( int size );

void phaseGrating(  cufftComplex* V_d, int nAt, int nZ, params_t* params, params_t* params_d, float* xyzCoordFP_d, float imPot, int* Z_d, int* Zlist, float* occ_d, int i );

int listOfElements( int* Zlist, int nAt, int *Z_d );

void setCufftPlanBatch( cufftHandle* plan, params_t* params );

void addNoiseAndMtf( cufftComplex* I_d, float dose, int k, curandState* poissonState, params_t* params, params_t* params_d );

void saveMeasurements( float* J_d, params_t* params );

void savePotential( cufftComplex* V_d, params_t* params );

float subSliceRatio( float slice, float subSlice );

void setSubSlices( params_t* params, params_t* params_d, float ratio );

void farFieldTransform( cufftComplex* psi, params_t* params, params_t* params_d );

void shiftCoordinates( float* xyzCoord_d, params_t* params, int scnI );

void integrateRecordings( float* J_h, cufftComplex* J_d, params_t* params, params_t* params_d, int k, int scnI );

#endif 