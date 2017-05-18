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
#include <curand.h>
#include "curand_kernel.h"
#include "paramStructure.h"
#include "projectedPotential.h"
#include "optimFunctions.h"

__global__ void incomingWave_d(cufftComplex* psi, int k, params_t* params);

__global__ void sphericalWavefront_d ( cufftComplex* psi, params_t* params_d );

__global__ void tiltBeam_d ( cufftComplex* psi, int k, params_t* params, int flag );

__global__ void condensedBeam_d ( cufftComplex* psi, params_t* params_d );

__global__ void taperedCosineWindow_d ( cufftComplex* psi, params_t* params, float bckgrnd );

__global__ void potential2Transmission(cufftComplex* V, int size);

__global__ void zeroHighFreq(cufftComplex* f, int dim1, int dim2 );

__global__ void fresnelPropagatorDevice(cufftComplex* frProp, params_t* params, int flag);

__global__ void shiftSpectiltBack(cufftComplex* psi, int k, params_t* params, int flag);

__global__ void multiplyLensFunction(cufftComplex* psi, int k, params_t* params_d, int flag);

__global__ void multiplyDC1Prefactor_d ( cufftComplex* psi, int k, params_t* params );

__global__ void multiplyDA1_0Prefactor_d ( cufftComplex* psi, int k, params_t* params );

__global__ void multiplyDA1_1Prefactor_d ( cufftComplex* psi, int k, params_t* params );

__global__ void multiplyDA2_0Prefactor_d ( cufftComplex* psi, int k, params_t* params );

__global__ void multiplyDA2_1Prefactor_d ( cufftComplex* psi, int k, params_t* params );

__global__ void multiplyDB2_0Prefactor_d ( cufftComplex* psi, int k, params_t* params );

__global__ void multiplyDB2_1Prefactor_d ( cufftComplex* psi, int k, params_t* params );

__global__ void multiplyDC3Prefactor_d ( cufftComplex* psi, int k, params_t* params );

__global__ void multiplySpatialFreqTilt_d ( cufftComplex* psi, const int k, params_t* params, const int xy );

__global__ void intensityValues(cufftComplex* psi, int m12);

__global__ void intensityValsInplc_d ( cufftComplex* psi, int m12 );

__global__ void multiplyMtf (cufftComplex* psi, params_t* params, int flag);

__global__ void multiplySpatialIncoherence(cufftComplex* psi, int k, params_t* params, int flag);

__global__ void zeroImagPart(cufftComplex* f, int size);

__global__ void multiplyComplexConjugate(cufftComplex* dEdV, cufftComplex* psi, int m12);

__global__ void copyMiddleOutComplex2D_d ( cuComplex* fOut, cuComplex* fIn, params_t* params );

__global__ void flattenDynamicRange( cuComplex* f, int size, int flag );

__global__ void tiltObject_d ( cuComplex* fOut, cuComplex* fIn, int it, params_t* params_d, int flag );


void rawMeasurement( cufftComplex *I_d, params_t* params, params_t* params_d, int k, int nAt, int nZ, float *xyzCoord_d, 
		    float imPot, int *Z_d, int *Zlist, float *DWF_d, float *occ_d, curandState *dwfState_d, int* itCnt );

void forwardPropagation(cufftComplex* psi, cufftComplex* V, cufftComplex* frProp, params_t* params, params_t* params_d );

void bandwidthLimit(cufftComplex* f, params_t* params_h);

void incomingWave( cufftComplex* psi, params_t* params, params_t* params_d, int k );

void fresnelPropagator(cufftComplex* frProp, params_t* params_h, params_t* params_d, int flag);

void convolveWithFrProp(cufftComplex* psi, cufftComplex* frProp, params_t* params);

void applyLensFunction(cufftComplex* psi, int k, params_t* params, params_t* params_d);

void applyMtfAndIncoherence(cufftComplex* psi, int k, params_t* params, params_t* params_d, int flag );

void saveTestImage(char* fileName, cufftComplex* src, int m12);

#endif