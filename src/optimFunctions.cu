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


/*__global__ void trialPotential ( cuComplex* V, params_t* params )
{
    const int m1 = params->IM.m1;
    const int m2 = params->IM.m2;
    const int m3 = params->IM.m3;
    V[m3 * m1 * m2 / 2 + m1 * m2 / 2 + m1 / 2].x = 2.0f;
    V[m3 * m1 * m2 / 2 + m1 * m2 / 2 + m1 / 2].y = 0.1f;
} */


/*__global__ void addConst ( cuComplex* dst, cuComplex cst, int size )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < 2 * size )
    {
        if ( ( i % 2 ) == 0 )
        { dst[i / 2].x += cst.x; }
        else
        { dst[i / 2].y += cst.y; }
    }
}*/


/*__global__ void flipPotentialDevice ( cuComplex* V, cuComplex thold, int size )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    {
		if ( V[i].x < thold.x )
		{    V[i].x *= -1.f; }
		if ( V[i].y < thold.y )
		{    V[i].y *= -1.f; }
	}
}*/

/*__global__ void flipPotentialAbsVals_d ( cuComplex* V, cuComplex thold, int size )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    {
		if ( absoluteValueOF( V[i].x ) < thold.x )
		{    V[i].x *= -1.f; }
		if ( absoluteValueOF( V[i].y ) < thold.y )
		{    V[i].y *= -1.f; }
	}
}*/


/*__global__ void applyThreshold_d ( cuComplex* V, cuComplex thold, int size )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < size )
    {
		if ( V[i].x < thold.x )
        {    V[i].x = 0.f; }
        if ( V[i].y < thold.y )
        {    V[i].y = 0.f; }
    }
}*/


__global__ void copyMiddleOut ( float* Imodel, cuComplex* I_d, params_t* params )
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int n1 = params->IM.n1;

    if ( j < n1 * params->IM.n2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, j, n1 );
        const int m1 = params->IM.m1;
        Imodel[j] = I_d[ i1 + params->IM.dn1 + m1 * ( i2 + params->IM.dn2 )].x;
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


/*__global__ void copyDefoci_d ( params_t* params, int k, float defocus_k )
{
	// Call it with gridsize = 1 and blocksize = 1.
	params->IM.defoci[k] = defocus_k;
}*/


/*__global__ void copyTiltspec_d ( params_t* params, int k, float t0, float t1 )
{
	// Call it with gridsize = 1 and blocksize = 1.
	params->IM.tiltspec[2*k] = t0;
	params->IM.tiltspec[2*k + 1] = t1;
}*/

/*__device__ float absoluteValueOF( float x )
{    return ( sqrtf( x * x + 1e-8f ) ); }*/


/*void randShuffle ( int* t, int n )
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
}*/


/*bool smallErrorDiff( float* E, int j )
{
	bool isSmall = false;

	const float eps = 1e-4f;
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
}*/


/*void initialUniformPDD( cufftComplex* f_d, float a, float b, int size )
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
}*/

/*void initialSparsePDD( cufftComplex* f_d, float b, params_t* params )
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

}*/

// void consistentPfParams (params_t* params )
// {
// // 	params->RE.pftr0_x = 0.01f;
// // 	params->RE.pftr0_y = 0.01f;
// // 
// // 	if ( params->RE.pftr < 0.001f )
// // 	{    params->RE.pftr = 0.001f; }
// // 
// // 	if ( params->RE.pftr > 0.999f )
// // 	{    params->RE.pftr = 0.999f; }
// }

void progressCounter(int j, int jTot)
{
    if ( (j == 0) && (jTot > 1) )
	{   fprintf(stderr, " Progress: 0%%");}

	const float frac = 100.f * ((float) j) / ((float) jTot);
	const float incr = 100.1f / ((float) jTot);

    for (float k = 1.f; k < 100.f; k++)
    {
        if ( (frac > k) && ( frac < ( k + incr ) ) )
		{
			if (k < 10.5f)
			{	fprintf(stderr, "\b\b");}
			else
			{	fprintf(stderr, "\b\b\b");}
            fprintf(stderr, "%i%%", (int) k);
		}
    }

    if (j == (jTot-1))
	{
		fprintf(stderr, "\b\b\b");
        fprintf(stderr, "%i%% ...\n", 100);
	}
}

void initialPotential ( cufftComplex* V_d, params_t* params )
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


