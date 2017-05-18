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

#ifndef complexMath_ljhbvdfuqzwtre
#define complexMath_ljhbvdfuqzwtre

#include <cufft.h>
#include <cuComplex.h>
#include "cuda_assert.hpp"
#include "float.h"
#include "coordArithmetic.h"


__global__ void realPartDevice (float* dst, cuComplex* src, int size);

__global__ void imagPartDevice (float* dst, cuComplex* src, int size);

__global__ void multiplyElementwise ( cufftComplex* f0, cufftComplex* f1, int size );

__global__ void initialValues ( cuComplex* V, int size, float initRe, float initIm );

__global__ void multiplyElementwiseFast ( cufftComplex* f0, cufftComplex* f1, int size );

__global__ void sumElements_d( cuComplex* dst, cuComplex* src, cuComplex thold, int sizeSrc, const int flag, const float mu0 );

__global__ void addDerivedAbsVals(cuComplex* dst, cuComplex* src, float fctr, int size, float mu0 );

__global__ void devideByAbsValue_d( cuComplex* f, int dim, float mu0 );

__global__ void myCaxpy ( cuComplex* y, cuComplex* x, cuComplex a, int size );

__device__ float absoluteValue( float x, const float mu0 );

__global__ void upperThreshold ( cufftComplex* f, int size, float thrRe, float thrIm );

__global__ void	lowerThreshold ( cufftComplex* f, int size, float thrRe, float thrIm );

__global__ void copyCufftShift_dim1_d ( cufftComplex* f0, cufftComplex* f1, int dim1, int dim2, int n0, int offSet );

__global__ void copyCufftShift_dim2_d ( cufftComplex* f0, cufftComplex* f1, int dim1, int dim2, int n0, int offSet );

__global__ void cufftShift2D_dim1_d ( cufftComplex* f1, cufftComplex* f0, int dim1, int dim2, int n0, int offSet );

__global__ void cufftShift2D_dim2_d ( cufftComplex* f1, cufftComplex* f0, int dim1, int dim2, int n0, int offSet );

__global__ void cufftIShift2D_dim1_d ( cufftComplex* f1, cufftComplex* f0, int dim1, int dim2, int n0, int offSet );

__global__ void cufftIShift2D_dim2_d ( cufftComplex* f1, cufftComplex* f0, int dim1, int dim2, int n0, int offSet );


void realPart(float* Vri, cufftComplex* V, int size);

void imagPart(float* Vri, cufftComplex* V, int size);

float sumElementsF( float* V, const int size );

float sumElements(cuComplex* V, const int size);

float sumAbsValElements(cuComplex* V, const int size, const float mu0 );

cuComplex sumThresElements( cuComplex* V, const int size, cuComplex thold );

cuComplex sumThresAbsElements( cuComplex* V, const int size, cuComplex thold );

cuComplex sumElementsComplex(cuComplex* V, const int size);

cuComplex sumElements_helper( cuComplex* V, cuComplex thold, const int size, const int flag, const float mu0 );

void cufftShift2D_h( cufftComplex* f0_d, int n1, int n2, int bS );

void cufftShiftI2D_h( cufftComplex* f0_d, int n1, int n2, int bS );

#endif