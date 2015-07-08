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

#ifndef multisliceSimulation_lqgflqzrfsdlkuhipoujerglhjkqefqebf
#define multisliceSimulation_lqgflqzrfsdlkuhipoujerglhjkqefqebf

#include <cufft.h>
#include "paramStructure.h"
#include "projectedPotential.h"
#include "crystalMaker.h"
#include <stdio.h>


__global__ void incomingWave_d(cufftComplex* psi, int k, params_t* params);

__global__ void tiltBeam_d ( cufftComplex* psi, int k, params_t* params, int flag );

__global__ void taperedCosineWindow_d ( cufftComplex* psi, params_t* params );

__global__ void smallCosineWindow ( cufftComplex* f, params_t* params );

__global__ void selectedAreaAperture ( cufftComplex* f, params_t* params );

__global__ void potential2Transmission(cufftComplex* t, cufftComplex* V, int size);

__global__ void zeroHighFreq(cufftComplex* f, int dim1, int dim2, float mindimsq);

__global__ void fresnelPropagatorDevice(cufftComplex* frProp, params_t* params );

__global__ void multiplyLensFunction(cufftComplex* psi, int k, params_t* params_d );

__global__ void intensityValues(cufftComplex* psi, int m12);

__global__ void multiplyMtf (cufftComplex* psi, params_t* params );

__global__ void multiplySpatialIncoherence(cufftComplex* psi, int k, params_t* params );

__global__ void multiplySpatialIncoherenceDP ( cufftComplex* psi, int k, params_t* params );

__global__ void zeroImagPart(cufftComplex* f, int size);

__global__ void copyMiddleOutComplex2D_d ( cuComplex* fOut, cuComplex* fIn, params_t* params );

__global__ void areaMask ( float* f, params_t* params );

__global__ void gaussianMask ( cufftComplex* f, params_t* params );


void forwardPropagation ( cufftComplex* psi, cufftComplex* V,  cufftComplex* frProp, cufftComplex* t, params_t* params, params_t* params_d );

void backwardPropagation(cufftComplex* dEdV, cufftComplex* dEdVtilt, cufftComplex* psi, cufftComplex* V, params_t* params, params_t* params_d, int k);

void derivativeFunction(cufftComplex* dEdV, float* dNuis, cufftComplex* psi, cufftComplex* V, cufftComplex* dEtdV, params_t* params, params_t* params_d, int k);

float derivativeDefocus( cufftComplex* dEdV, cufftComplex* psi, params_t* params, params_t* params_d, int k );

float derivativeTilts( cufftComplex* dEdV, const int xy, cufftComplex* psi, params_t* params, params_t* params_d, int k );

void bandwidthLimit(cufftComplex* f, params_t* params_h);

void incomingWave( cufftComplex* psi, int k, params_t* params, params_t* params_d );

void fresnelPropagator(cufftComplex* frProp, params_t* params_h, params_t* params_d );

void convolveWithFrProp(cufftComplex* psi, cufftComplex* frProp, params_t* params);

void  applyLensFunction ( cufftComplex* psi, int k, params_t* params, params_t* params_d );

void saveTestImage(char* fileName, cufftComplex* src, int m12);

void macroTest(int m);

#endif