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

#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include "paramStructure.h"
#include "rwBinary.h"
#include "optimFunctions.h"
#include "complexMath.h"
#include "cuda_assert.hpp"
#include "cublas_assert.hpp"
#include <cublas_v2.h>
#include <cuComplex.h>
#include "multisliceSimulation.h"
#include "coordArithmetic.h"
#include "performanceTimer.h"
#include <float.h>
#include <cmath>
#include "projectedPotential.h"
#include "globalVariables.h"


__global__ void trialPotential ( cuComplex* V, params_t* params )
{
    const int m1 = params->IM.m1;
    const int m2 = params->IM.m2;
    const int m3 = params->IM.m3;
    V[m3 * m1 * m2 / 2 + m1 * m2 / 2 + m1 / 2].x = 2.0f;
    V[m3 * m1 * m2 / 2 + m1 * m2 / 2 + m1 / 2].y = 0.1f;
}


__global__ void addConst ( cuComplex* dst, cuComplex cst, int size )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < 2 * size )
    {
        if ( ( i % 2 ) == 0 )
        { dst[i / 2].x += cst.x; }
        else
        { dst[i / 2].y += cst.y; }
    }
}


__global__ void flipPotentialDevice ( cuComplex* V, cuComplex thold, int size )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    {
		if ( V[i].x < thold.x )
		{    V[i].x *= -1.f; }
		if ( V[i].y < thold.y )
		{    V[i].y *= -1.f; }
	}
}

__global__ void flipPotentialAbsVals_d ( cuComplex* V, cuComplex thold, int size )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    {
		if ( absoluteValueOF( V[i].x ) < thold.x )
		{    V[i].x *= -1.f; }
		if ( absoluteValueOF( V[i].y ) < thold.y )
		{    V[i].y *= -1.f; }
	}
}


__global__ void applyThreshold_d ( cuComplex* V, cuComplex thold, int size )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < size )
    {
		if ( V[i].x < thold.x )
        {    V[i].x = 0.f; }
        if ( V[i].y < thold.y )
        {    V[i].y = 0.f; }
    }
}


__global__ void copyMiddleOut ( float* Imodel, cuComplex* psi, params_t* params )
{
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int n1 = params->IM.n1;

	if ( j < n1 * params->IM.n2 ) {
		int i1, i2;
		dbCoord ( i1, i2, j, n1 );
		const int m1 = params->IM.m1;
		Imodel[j] = psi[m1 * params->IM.m2 * ( params->IM.m3 + 3 ) + i1 + params->IM.dn1 + m1 * ( i2 + params->IM.dn2 )].x;
	}
}


__global__ void copyMiddleIn ( cuComplex* dE, float* Imodel, params_t* params )
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int n1 = params->IM.n1;

    if ( j < n1 * params->IM.n2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, j, n1 );
        const int m1 = params->IM.m1;
        dE[m1 * params->IM.m2 * ( params->IM.m3 + 3 ) + i1 + params->IM.dn1 + m1 * ( i2 + params->IM.dn2 )].x = Imodel[j];
    }
}


__global__ void copyDefoci_d ( params_t* params, int k, float defocus_k )
{
	// Call it with gridsize = 1 and blocksize = 1.
	params->IM.defoci[k] = defocus_k;
}


__global__ void copyTiltspec_d ( params_t* params, int k, float t0, float t1 )
{
	// Call it with gridsize = 1 and blocksize = 1.
	params->IM.tiltspec[2*k] = t0;
	params->IM.tiltspec[2*k + 1] = t1;
}

__device__ float absoluteValueOF( float x )
{    return ( sqrtf( x * x + 1e-8f ) ); }


__global__ void limitDerivatives_d ( cufftComplex* dEdV, cufftComplex* V, int size, float thrLo, float thrHi )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    {
        if ( ( V[i].y < thrLo ) && ( dEdV[i].y > 0.f ) )
		{
			dEdV[i].x = 0.f;
			dEdV[i].y = 0.f;
		}
		if ( ( V[i].y > thrHi ) && ( dEdV[i].y < 0.f ) )
		{
			dEdV[i].x = 0.f;
			dEdV[i].y = 0.f;
		}
    }
}

__global__ void signValues_d ( float* f, int size )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	float eps = 1e-9f;

    if ( i < size )
	{    
		if ( f[i] < -eps )
		{    f[i] = -1.f; }
		else { if ( f[i] > eps )
		{    f[i] = 1.f; }
		else
		{    f[i] = 0.f; } }
	}
}

__global__ void ringFilter_d ( cufftComplex* f, int dim1, int dim2 )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i < dim1 * dim2 )
	{	  
		int i1, i2;

		f[i].x = 0.f;
		f[i].y = 0.f;

		dbCoord ( i1, i2, i, dim1 );
		iwCoordIp ( i1, dim1 );
		iwCoordIp ( i2, dim2 );

		if ( i1 == -1 )
        {
            if ( i2 == -1 )
            {    f[i].x =      -0.037608f ; }
            if ( i2 == 0 )
            {    f[i].x =      -0.249254f ; }
            if ( i2 == 1 )
            {    f[i].x =      -0.037608f ; }
        }

        if ( i1 == 0 )
        {
            if ( i2 == -1 )
            {    f[i].x =      -0.249254f ; }
            if ( i2 == 0 )
            {    f[i].x =       1.147448f ; }
            if ( i2 == 1 )
            {    f[i].x =      -0.249254f ; }
        }

        if ( i1 == 1 )
        {
            if ( i2 == -1 )
            {    f[i].x =      -0.037608f ; }
            if ( i2 == 0 )
            {    f[i].x =      -0.249254f ; }
            if ( i2 == 1 )
            {    f[i].x =      -0.037608f ; }
        }

		f[i].x /= ( (float) ( dim1 * dim2 ) );
	}
}

__global__ void copySingle2Double ( double* fD, float* fS,  int size )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
	{    fD[i] = (double) fS[i];	}
}

__global__ void divideBySqrt_d ( float* f0, float* f1, int size )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const float eps = 1e-2f;

    if ( i < size )
	{    f0[i] /= sqrtf( fabsf( f1[i] ) + eps ); }
}

void randShuffle ( int* t, int n )
{
    int c, s;

    for ( int i = 0; i < n - 1; i++ )
    {
        c =  rand () / ( RAND_MAX / ( n - i ) + 1 );
        // swap
        s = t[i];
        t[i] = t[i + c];
        t[i + c] = s;
    }
}


bool smallErrorDiff( float* E, int j )
{
	bool isSmall = false;

	const float eps = 1e-6f;
	float varE = 0.f;
	float temp;

	if ( j > 2 )
	{
		temp = E[j - 3] - E[j - 2];
		varE += temp * temp;
		temp = E[j - 2] - E[j - 1];
		varE += temp * temp;
		temp = E[j - 1] - E[j];
		varE = 0.33333333f * ( varE + temp * temp );
		if ( varE < ( eps * eps ) )
		{    
			isSmall = true; 
			fprintf ( stderr, "\n  Three consecutive function values differ by less than %g; exiting at iteration %i...\n", eps, j );
		}
	}

	return ( isSmall );
}


void initialUniformPDD( cufftComplex* f_d, float a, float b, int size )
{
	// Initializes f_d with uniform random values between a and b, a < b.

	cufftComplex* f_h;
	float randMaxf;

	f_h = ( cufftComplex* ) malloc ( size * sizeof ( cufftComplex ) );
	randMaxf = (float) RAND_MAX;

	b -= a;
	for ( int j = 0; j < size; j++ )
	{
		f_h[j].x = ( ( (float) rand() ) / randMaxf ) * b + a;
		f_h[j].y = ( ( (float) rand() ) / randMaxf ) * b + a;
	}

	cuda_assert ( cudaMemcpy ( f_d, f_h, size * sizeof ( cufftComplex ), cudaMemcpyHostToDevice ) );

	free( f_h );
}

void initialSparsePDD( cufftComplex* f_d, float b, params_t* params )
{
	// Initializes f_d with random values between 0 and b. The mean of this distribution equals 8*b times the density of FCC Au.

	cufftComplex* f_h;
	float randMaxf, alpha;
	int size = params->IM.m1 * params->IM.m2 * params->IM.m3;

	f_h = ( cufftComplex* ) malloc ( size * sizeof ( cufftComplex ) );
	randMaxf = (float) RAND_MAX;
	alpha = 1.f / ( 5.0e29f * ( params->IM.d1 * params->IM.d2 * params->IM.d3 ) ) - 1.f; // is about 14

	for ( int j = 0; j < size; j++ )
	{
		f_h[j].x = powf( ( ( (float) rand() ) / randMaxf ), alpha ) * b;
		f_h[j].y = powf( ( ( (float) rand() ) / randMaxf ), alpha ) * b;
	}

	cuda_assert ( cudaMemcpy ( f_d, f_h, size * sizeof ( cufftComplex ), cudaMemcpyHostToDevice ) );

	free( f_h );

}

void saveResults ( cufftComplex* V_d, float* E, float* L1, int j, params_t* params_d, int n3 )
{
 	params_t* params;
	allocParams ( &params, n3 );
	getDeviceParams ( &params, &params_d, n3 );
	setCufftPlan ( params );
	setCublasHandle ( params );
	
	const int m123 = params->IM.m1 * params->IM.m2 * params->IM.m3;
	cufftComplex* V_h;
	V_h = ( cufftComplex* ) malloc ( m123 * sizeof ( cufftComplex ) );
	cuda_assert ( cudaMemcpy ( V_h, V_d, m123 * sizeof ( cufftComplex ), cudaMemcpyDeviceToHost ) );
	
	float* Vri;
	Vri = ( float* ) calloc ( m123, sizeof ( float ) );

	realPart ( Vri, V_h, m123 );
	writeBinary ( "PotentialReal.bin", Vri, m123 );

	imagPart ( Vri, V_h, m123 );
	writeBinary ( "PotentialImag.bin", Vri, m123 );
	
	free ( Vri );	
	free ( V_h );
	freeParams ( &params );
} 

int progressCounter(int j, params_t* params )
{
	int kR = 0, kP = 0;
	int jTot = 0;
	j += 1;
	
	jTot = params->IM.frPh; // Calculate total number of iterations
	if ( jTot < 1 )	{   
		jTot = 1; 
	}
	jTot *= params->SCN.o1 * params->SCN.o2 * params->IM.n3 * params->IM.m3;

	kP = (int) roundf( ( ( (float) ( j - 1 ) ) / ( (float) jTot ) ) * 100.f );
	kR = (int) roundf( ( ( (float) j ) / ( (float) jTot ) ) * 100.f );
	if ( kP !=  kR ) { // Only print when there's a new percentage
		if ( kR < 100 ) { // print three dots at 100% to indicate it's almost ready, but not quite yet
			fprintf(stderr, "\r  Progress: %i%%", kR );
		} else {
			fprintf(stderr, "\r  Progress: 100%% ..." );
		}
	}
	if ( j == jTot ) {
		fprintf(stderr, "\n" );
	}
	return( j );
}

void readInitialPotential ( cufftComplex* V_d, params_t* params )
{
    float* Vri;
    cufftComplex* V_h;
    const int size = params->IM.m1 * params->IM.m2 * params->IM.m3;
    Vri = ( float* ) malloc ( size * sizeof ( float ) );
    V_h = ( cufftComplex* ) malloc ( size * sizeof ( cufftComplex ) );

    readBinary ( "PotentialInitReal.bin", Vri, size );

    for ( int j = 0; j < size; j++ )
    { V_h[j].x = Vri[j]; }

    readBinary ( "PotentialInitImag.bin", Vri, size );

    for ( int j = 0; j < size; j++ )
    { V_h[j].y = Vri[j]; }

    cuda_assert ( cudaMemcpy ( V_d, V_h, size * sizeof ( cufftComplex ), cudaMemcpyHostToDevice ) );

    free ( Vri );
    free ( V_h );
}


void readIncomingWave ( cufftComplex* V_d, int k, params_t* params )
{
    float* Vri;
    cufftComplex* V_h;
    const int size = params->IM.m1 * params->IM.m2;
    Vri = ( float* ) malloc ( size * sizeof ( float ) );
    V_h = ( cufftComplex* ) malloc ( size * sizeof ( cufftComplex ) );

	if ( k == 0 )
	{    readBinary ( "Illumination_real_1.bin", Vri, size ); }
	if ( k == 1 )
	{    readBinary ( "Illumination_real_2.bin", Vri, size ); }
	if ( k == 2 )
	{    readBinary ( "Illumination_real_3.bin", Vri, size ); }
	if ( k == 3 )
	{    readBinary ( "Illumination_real_4.bin", Vri, size ); }
	if ( k == 4 )
	{    readBinary ( "Illumination_real_5.bin", Vri, size ); }
	if ( k == 5 )
	{    readBinary ( "Illumination_real_6.bin", Vri, size ); }
	if ( k == 6 )
	{    readBinary ( "Illumination_real_7.bin", Vri, size ); }
	if ( k == 7 )
	{    readBinary ( "Illumination_real_8.bin", Vri, size ); }
	if ( k == 8 )
	{    readBinary ( "Illumination_real_9.bin", Vri, size ); }
	if ( k == 9 )
	{    readBinary ( "Illumination_real_10.bin", Vri, size ); }

    for ( int j = 0; j < size; j++ )
    { V_h[j].x = Vri[j]; }

    if ( k == 0 )
	{    readBinary ( "Illumination_imag_1.bin", Vri, size ); }
	if ( k == 1 )
	{    readBinary ( "Illumination_imag_2.bin", Vri, size ); }
	if ( k == 2 )
	{    readBinary ( "Illumination_imag_3.bin", Vri, size ); }
	if ( k == 3 )
	{    readBinary ( "Illumination_imag_4.bin", Vri, size ); }
	if ( k == 4 )
	{    readBinary ( "Illumination_imag_5.bin", Vri, size ); }
	if ( k == 5 )
	{    readBinary ( "Illumination_imag_6.bin", Vri, size ); }
	if ( k == 6 )
	{    readBinary ( "Illumination_imag_7.bin", Vri, size ); }
	if ( k == 7 )
	{    readBinary ( "Illumination_imag_8.bin", Vri, size ); }
	if ( k == 8 )
	{    readBinary ( "Illumination_imag_9.bin", Vri, size ); }
	if ( k == 9 )
	{    readBinary ( "Illumination_imag_10.bin", Vri, size ); }

    for ( int j = 0; j < size; j++ )
    { V_h[j].y = Vri[j]; }

    cuda_assert ( cudaMemcpy ( V_d, V_h, size * sizeof ( cufftComplex ), cudaMemcpyHostToDevice ) );

    free ( Vri );
    free ( V_h );
}


bool isPowOfTwo ( int x )
{
	return ( (x > 0) && !(x & (x - 1)) );
}


bool isPowOfHalf( int it )
{
	bool x = false;
	int I;
	
	if ( isPowOfTwo( it ) )
	{    x = true; }
	else
	{
		I = (int) roundf( powf( 2.f, floorf( log10f( (float) it ) / 0.30103000f ) + 0.5f ) );
		if ( it == I )
		{    x = true; }
	}

	return( x );
}
