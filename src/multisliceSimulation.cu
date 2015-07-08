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

#include "multisliceSimulation.h"
#include <stdio.h>
#include <stdlib.h>
#include "coordArithmetic.h"
#include <cufft.h>
#include "paramStructure.h"
#include "cuda_assert.hpp"
#include <cuComplex.h>
#include <math.h>
#include "complexMath.h"
#include "cufft_assert.h"
#include "cublas_assert.hpp"
#include "rwBinary.h"
#include "performanceTimer.h"


__global__ void potential2Transmission ( cufftComplex* t, cufftComplex* V, int size )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    {
        float Vx = V[i].x;
        float Vy = V[i].y;
        t[i].x = expf ( -Vy ) * cosf ( Vx );
        t[i].y = expf ( -Vy ) * sinf ( Vx );
    }
}


__global__ void incomingWave_d ( cufftComplex* psi, int k, params_t* params )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

    if ( i < dim1 * dim2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, i, dim1 );
        owCoordIp ( i1, dim1 );
        owCoordIp ( i2, dim2 );

        if ( ( ( ( float ) ( i1 * i1 ) ) / ( ( float ) ( dim1 * dim1 ) ) + ( ( float ) ( i2 * i2 ) ) / ( ( float ) ( dim2 * dim2 ) ) ) < 0.25f ) //??
        {
        float x2 = params->EM.lambda;
	    float x1 = ( ( float ) i1 ) * ( params->IM.d1 / x2 ) * params->IM.tiltbeam[2 * k + 1];

            x2 = ( ( float ) i2 ) * ( params->IM.d2 / x2 ) * params->IM.tiltbeam[2 * k]; //recycle x2

            x1 = 2.f * params->cst.pi * ( x1 + x2 ); // recycle x1
            psi[i].x = cosf ( x1 );
            psi[i].y = sinf ( x1 );
        }

        else
        {
            psi[i].x = 0.f;
            psi[i].y = 0.f;
        }
    }
}


__global__ void tiltBeam_d ( cufftComplex* psi, int k, params_t* params, int flag )
{
	// Multiply with a phase ramp
	// flag == +1: add beamtilt to the beam direction.
	// flag == -1: remove beamtilt from the beam direction.

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

	if ( i < dim1 * dim2 )
	{
		int i1, i2;
		float x1, x2;
		dbCoord ( i1, i2, i, dim1 );
		owCoordIp ( i1, dim1 );
		owCoordIp ( i2, dim2 );

		x2 = params->EM.lambda * ( (float) flag );
		x1 = ( ( float ) i1 ) * ( params->IM.d1 / x2 ) * params->IM.tiltbeam[2 * k + 1];
		x2 = ( ( float ) i2 ) * ( params->IM.d2 / x2 ) * params->IM.tiltbeam[2 * k]; //recycle x2
		x1 = 2.f * params->cst.pi * ( x1 + x2 ); // recycle x1
		x2 = sinf ( x1 ); // imaginary part
		x1 = cosf ( x1 ); // real part

		float temp = psi[i].x;

		// Apply the phase ramp
		psi[i].x = x1 * temp - x2 * psi[i].y;  // real*real - imag*imag
		psi[i].y = x2 * temp + x1 * psi[i].y;  // real*imag + imag*real
	}
}


__global__ void taperedCosineWindow_d ( cufftComplex* psi, params_t* params )
{
	// Apply the tapered cosine or Tukey window: http://en.wikipedia.org/wiki/Window_function

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int dim1 = params->IM.m1;
	const int dim2 = params->IM.m2;

	if ( i < dim1 * dim2 )
	{
		int i1, i2;
		float w, alpha, x;
		dbCoord ( i1, i2, i, dim1 );

		w = 1.f;

		alpha = 2.f * ( ( (float) params->IM.dn1 ) / ( (float) dim1 ) ); // alpha parameter of the tapered cosine window
		x = ( (float) i1 ) / ( (float) ( dim1 - 1 ) );
		if ( x < alpha * 0.5f )
		{    w = 0.5f * ( 1.f + cosf( params->cst.pi * ( 2.f * x / alpha - 1.f ) ) ); }
		else { if ( x > 1.f - 0.5f * alpha )
		{    w = 0.5f * ( 1.f + cosf( params->cst.pi * ( 2.f * x / alpha + 1.f - 2.f / alpha ) ) ); } }

		alpha = 2.f * ( ( (float) params->IM.dn2 ) / ( (float) dim2 ) ); // alpha parameter of the tapered cosine window
		x = ( (float) i2 ) / ( (float) ( dim2 - 1 ) );
		if ( x < alpha * 0.5f )
		{    w *= 0.5f * ( 1.f + cosf( params->cst.pi * ( 2.f * x / alpha - 1.f ) ) ); }
		else { if ( x > 1.f - 0.5f * alpha )
		{    w *= 0.5f * ( 1.f + cosf( params->cst.pi * ( 2.f * x / alpha + 1.f - 2.f / alpha ) ) ); } }

		psi[i].x *= w;
		psi[i].y *= w;
	}
}


__global__ void smallCosineWindow ( cufftComplex* f, params_t* params )
{
	// Apply a cosine window to the inner n1xn2 region of f, everything outside of this region is set to 0.

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int dim1 = params->IM.m1;
	const int dim2 = params->IM.m2;

	if ( i < dim1 * dim2 )
	{
		int i1, i2;
		float w = 0.f, x;

		dbCoord ( i1, i2, i, dim1 );
		owCoordIp ( i1, dim1 );
		owCoordIp ( i2, dim2 );

		x = ( (float) i1 ) / ( (float) ( params->IM.n1 ) );
		if ( fabsf( x ) < 0.5f )
		{    w = 0.5f * ( cosf( params->cst.pi * x * 2.f ) + 1.f ); }
		
		x = ( (float) i2 ) / ( (float) ( params->IM.n2 ) );
		if ( fabsf( x ) < 0.5f )
		{    w *= 0.5f * ( cosf( params->cst.pi * x * 2.f) + 1.f ); }
		else
		{    w = 0.f; }

		f[i].x *= w;
		f[i].y *= w;
	}
}


__global__ void selectedAreaAperture ( cufftComplex* f, params_t* params )
{
	// Apply a cosine window to the inner n1xn2 region of f, everything outside of this region is set to 0.

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int dim1 = params->IM.m1;
	const int dim2 = params->IM.m2;

	if ( i < dim1 * dim2 )
	{
		int i1, i2;
		float w = 0.f, x, rSq;

		dbCoord ( i1, i2, i, dim1 );
		owCoordIp ( i1, dim1 );
		owCoordIp ( i2, dim2 );

		x = ( (float) i1 ) / ( (float) ( params->IM.n1 ) );
		x *= x;
		rSq = ( (float) i2 ) / ( (float) ( params->IM.n2 ) );
		rSq *= rSq;
		rSq += x;

		if ( rSq < 0.25f )
		{    w = 1.f; }
		

		f[i].x *= w;
		f[i].y *= w;
	}
}


__global__ void zeroHighFreq ( cufftComplex* f, int dim1, int dim2 )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float mindim = ( float ) dim1;

    if ( ( float ) dim2 < mindim )
    { mindim = ( float ) dim2; }

    if ( i < 2 * dim1 * dim2 )
    {
        int i0 = i / 2;
        int i1, i2;
        dbCoord ( i1, i2, i0, dim1 );
        iwCoordIp ( i1, dim1 );
        iwCoordIp ( i2, dim2 );

        if ( ( ( float ) ( i1 * i1 + i2 * i2 ) * 9.f / ( mindim * mindim ) ) > 1.f )
        {
            if ( ( i % 2 ) == 0 )
            { f[i0].x = 0.f; }

            else
            { f[i0].y = 0.f; }
        }
    }
}


__global__ void fresnelPropagatorDevice ( cufftComplex* frProp, params_t* params ) 
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

    if ( i < dim1 * dim2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, i, dim1 );
        iwCoordIp ( i1, dim1 );
        iwCoordIp ( i2, dim2 );

        float d3 = params->IM.d3;
		const float t1 = ( ( float ) ( i1 ) / ( ( float )dim1 ) ) * ( d3 / params->IM.d1 );
        const float t2 = ( ( float ) ( i2 ) / ( ( float )dim2 ) ) * ( d3 / params->IM.d2 );
        d3 = params->EM.lambda / d3;
		d3 = -params->cst.pi * ( t1 * t1 + t2 * t2 ) * d3;
        frProp[i].x = cosf ( d3 );
        frProp[i].y = sinf ( d3 );
    }
}


__global__ void multiplyLensFunction ( cufftComplex* psi, int k, params_t* params )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

    if ( i < dim1 * dim2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, i, dim1 );
        iwCoordIp ( i1, dim1 );
        iwCoordIp ( i2, dim2 );
        i2 = -i2; // row index points UP now.

        float nu  = ( ( ( float ) i1 ) / ( ( float ) dim1 ) ) * ( params->EM.lambda / params->IM.d1 );
        float nu2 = ( ( ( float ) i2 ) / ( ( float ) dim2 ) ) * ( params->EM.lambda / params->IM.d2 );
        float phi = atan2f ( nu2, nu );
        nu = sqrtf ( nu * nu + nu2 * nu2 );

        if ( nu < params->EM.ObjAp )
        {
            float W = nu * nu * ( 0.5f * (
                                      params->EM.aberration.A1_0 * cosf ( 2.f * ( phi - params->EM.aberration.A1_1 ) )
                                      + params->EM.aberration.C1_0 + params->IM.defoci[k] )
                                  + nu * ( 1.f / 3.f * (
                                               params->EM.aberration.A2_0 * cosf ( 3.f * ( phi - params->EM.aberration.A2_1 ) )
                                               + params->EM.aberration.B2_0 * cosf ( phi - params->EM.aberration.B2_1 ) )
                                           + nu * ( 0.25f * (
                                                   params->EM.aberration.A3_0 * cosf ( 4.f * ( phi - params->EM.aberration.A3_1 ) )
                                                   + params->EM.aberration.S3_0 * cosf ( 2.f * ( phi - params->EM.aberration.S3_1 ) )
                                                   + params->EM.aberration.C3_0 )
                                                    + nu * ( 0.2f * (
                                                            params->EM.aberration.A4_0 * cosf ( 5.f * ( phi - params->EM.aberration.A4_1 ) )
                                                            + params->EM.aberration.B4_0 * cosf ( phi - params->EM.aberration.B4_1 )
                                                            + params->EM.aberration.D4_0 * cosf ( 3.f * ( phi - params->EM.aberration.D4_1 ) ) )
                                                            + nu * ( 1.f / 6.f * (
                                                                    params->EM.aberration.A5_0 * cosf ( 6.f * ( phi - params->EM.aberration.A5_1 ) )
                                                                    + params->EM.aberration.R5_0 * cosf ( 4.f * ( phi - params->EM.aberration.R5_1 ) )
                                                                    + params->EM.aberration.S5_0 * cosf ( 2.f * ( phi - params->EM.aberration.S5_1 ) )
                                                                    + params->EM.aberration.C5_0 )
                                                                   ) ) ) ) ); // CLOSE ALL THE BRACKETS!

            // Recycle the variables:
            nu2 = params->EM.lambda;
            float damp = 1.f;
			if( params->IM.mode == 0 )
			{	
				damp = params->EM.defocspread * nu * nu / nu2;
				damp = expf ( -2.f * damp * damp );
			}
            nu = params->cst.pi;
            phi  = damp * cosf (  2.f * nu * ( W / nu2 ) ); // real part of CTF
            damp = damp * sinf ( -2.f * nu * ( W / nu2 ) ); // imag part of CTF
            nu  = psi[i].x; // real part of psi
            nu2 = psi[i].y; // imag part of psi

            psi[i].x = phi * nu - damp * nu2; // real*real - imag*imag
            psi[i].y = phi * nu2 + damp * nu; // real*imag + imag*real
        }

        else
        {
            psi[i].x = 0.f;
            psi[i].y = 0.f;
        }
    }
}


__global__ void intensityValues ( cufftComplex* psi, int m12 )
{
    // Takes the values of one layer of the object and writes the intensity.

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < m12 )
    {
        float psiRe = psi[i].x;
        float psiIm = psi[i].y;
        psi[i].x = psiRe * psiRe + psiIm * psiIm;
        psi[i].y = 0.f;
    }
}


__global__ void multiplyMtf ( cufftComplex* psi, params_t* params )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

    if ( i < dim1 * dim2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, i, dim1 );
        iwCoordIp ( i1, dim1 );
        iwCoordIp ( i2, dim2 );
             
       float nu1 = ( ( float ) i1 ) / ( ( float ) dim1 );
        float nu2 = ( ( float ) i2 ) / ( ( float ) dim2 );
       float mtf = sqrtf( nu1 * nu1 + nu2 * nu2 );

       mtf = ( params->EM.mtfa * expf ( -params->EM.mtfc * mtf ) + params->EM.mtfb * expf ( -params->EM.mtfd * mtf * mtf ) );

       nu1 *= params->cst.pi;
       nu2 *= params->cst.pi;
       mtf *= ( ( sinf ( nu1 ) + FLT_EPSILON ) / ( nu1 + FLT_EPSILON ) ) * ( ( sinf ( nu2 ) + FLT_EPSILON ) / ( nu2 + FLT_EPSILON ) );

        psi[i].x *= mtf;
        psi[i].y *= mtf;
    }
}


__global__ void multiplySpatialIncoherence ( cufftComplex* psi, int k, params_t* params )
{
	// Does the spatial incoherence for real-space data
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

    if ( i < dim1 * dim2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, i, dim1 );
        iwCoordIp ( i1, dim1 );
        iwCoordIp ( i2, dim2 );

        float damp = params->EM.lambda;
        float nusq = ( ( ( float ) i1 ) / ( ( float ) dim1 ) ) * ( damp / params->IM.d1 );
        damp = ( ( ( float ) i2 ) / ( ( float ) dim2 ) ) * ( damp / params->IM.d2 );
        nusq = nusq * nusq + damp * damp;
        damp = params->cst.pi * params->EM.illangle * params->IM.defoci[k];
        damp = expf ( -nusq * damp * damp );
        psi[i].x *= damp;
        psi[i].y *= damp;
    }
}


__global__ void multiplySpatialIncoherenceDP ( cufftComplex* psi, int k, params_t* params )
{
	// Does the spatial incoherence for diffraction data
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

    if ( i < dim1 * dim2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, i, dim1 );
        iwCoordIp ( i1, dim1 );
        iwCoordIp ( i2, dim2 );

        float x1 = ( (float) i1 ) * params->IM.d1;
		float x2 = ( (float) i2 ) * params->IM.d2;

		x1 = x1 * x1 + x2 * x2;
		x2 = params->cst.pi * params->EM.illangle / params->EM.lambda;

		x1 = expf( - x2 * x2 * x1  );

        psi[i].x *= x1;
        psi[i].y *= x1;
    }
}


__global__ void zeroImagPart ( cufftComplex* f, int size )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    { f[i].y = 0.f; }
}

__global__ void copyMiddleOutComplex2D_d ( cuComplex* fOut, cuComplex* fIn, params_t* params )
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int n1 = params->IM.n1;

    if ( j < n1 * params->IM.n2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, j, n1 );
		i1 = i1 + params->IM.dn1 + params->IM.m1 * ( i2 + params->IM.dn2 );
        fOut[j].x = fIn[i1].x;
		fOut[j].y = fIn[i1].y;
    }
}

__global__ void areaMask ( float* f, params_t* params )
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int dim1 = params->IM.m1;
	const int dim2 = params->IM.m2;
	
	
	const int dn1   = params->IM.dn1;
	const int dn2   = params->IM.dn2;

	if ( i < dim1 * dim2 )
	{
	  
	  
		int i1, i2;
		float w = 1.0f;

		dbCoord ( i1, i2, i, dim1 );
		
		if(i1 >  dn1-1 && i1< dim1-dn1  && i2 > dn2-1 && i2< dim2- dn2 )//1
		{    w *= 1.0f; }		
		__syncthreads ();
	
		if(i1 <= dn1-1 )
		{    w *= 0.5f * ( 1 - cos( 3.1415927f * (float) i1 / (float) dn1 ) ); }
		__syncthreads ();

		if(i1 >= dim1- dn1)
		{    w *= 0.5f * ( 1 - cos( 3.1415927f * (float)( dim1 - i1 ) / (float) dn1 ) ); }
		__syncthreads ();
		
		if(i2 <= dn2-1 )
		{    w *= 0.5f * ( 1 - cos( 3.1415927f * (float) i2 / (float) dn2 ) ); }
		__syncthreads ();
		
		if(i2 >= dim2- dn2)
		{    w *= 0.5f * ( 1 - cos( 3.1415927f * (float) ( dim2 - i2 ) / (float) dn2 ) ); }
		__syncthreads ();
		
		f[i] = w;
	}
}


__global__ void gaussianMask ( cufftComplex* f, params_t* params )
{
  
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int dim1 = params->IM.m1;
	const int dim2 = params->IM.m2;

	if ( i < dim1 * dim2 )
	{
		float gaussWidth = 100.0f;
		float w1,w2;
		float gw2 = gaussWidth*gaussWidth;

		int i1, i2;
		dbCoord ( i1, i2, i, dim1 );
		owCoordIp ( i1, dim1 );
		owCoordIp ( i2, dim2 );
		w1 = (float) i1 * gaussWidth / ( (float) dim1 ) * (float) i1 * gaussWidth / ( (float) dim1 );
		w2 = (float) i2 * gaussWidth / ( (float) dim2 ) * (float) i2 * gaussWidth / ( (float) dim2 ); 
		f[i].x = exp( - gw2 * ( w1 + w2 ) ); 
		f[i].y = 0.f;
	}
}


void forwardPropagation ( cufftComplex* psi, cufftComplex* V,  cufftComplex* frProp,  cufftComplex* t, params_t* params, params_t* params_d )
{
    const int m12 = params->IM.m1 * params->IM.m2;
    const int gS = params->CU.gS2D;
    const int bS = params->CU.bS;
    fresnelPropagator ( frProp, params, params_d );
    potential2Transmission <<< gS, bS>>> ( t, V, m12 ); // transform potential to transmission
    bandwidthLimit ( t, params );
    multiplyElementwise <<< gS, bS>>> ( t,  psi, m12 ); // do the transmission
    convolveWithFrProp ( t, frProp, params ); // do the propagation
    cublas_assert ( cublasCcopy ( params->CU.cublasHandle, m12, t, 1, psi, 1 ) );
}


void bandwidthLimit ( cufftComplex* f, params_t* params )
{
    cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, f, f, CUFFT_FORWARD ) );
    zeroHighFreq <<< 2 * params->CU.gS2D, params->CU.bS >>> ( f, params->IM.m1, params->IM.m2 );
    cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, f, f, CUFFT_INVERSE ) );
    const int m12 = params->IM.m1 * params->IM.m2;
    const float alpha = 1.f / ( ( float ) m12 );
    cublas_assert ( cublasCsscal ( params->CU.cublasHandle, m12, &alpha, ( cuComplex* ) f, 1 ) );
}


void incomingWave( cufftComplex* psi, int k, params_t* params, params_t* params_d )
{
	const int gS = params->CU.gS2D;
	const int bS = params->CU.bS;
	const int m12 = params->IM.m1 * params->IM.m2;
	float alpha = 1.f;

	initialValues <<< 2 * gS, bS >>> ( psi, m12, 1.f, 0.f );

	if ( ( params->IM.mode == 2) )
	{
		multiplyLensFunction <<< gS, bS >>> ( psi, k, params_d );
		cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, psi, psi, CUFFT_INVERSE ) );
		cufftShift2D_h( psi, params->IM.m1, params->IM.m2, bS );
		bandwidthLimit ( psi, params );
		cublas_assert( cublasScnrm2( params->CU.cublasHandle, m12, psi, 1, &alpha ) );
		alpha = sqrtf( (float) ( params->IM.n1 * params->IM.n2 ) ) / alpha; 
		cublas_assert( cublasCsscal( params->CU.cublasHandle, m12, &alpha, psi, 1 ) );
	}

	if ( params->IM.doBeamTilt )
	{    tiltBeam_d <<< params->CU.gS, params->CU.bS >>> ( psi, k, params_d, 1 ); }

	if( params->IM.doBeamTilt && ( ( params->IM.mode == 0 ) || ( params->IM.mode == 1 ) ) )
	{    
		taperedCosineWindow_d <<< params->CU.gS, params->CU.bS >>> ( psi, params_d );
		bandwidthLimit ( psi, params );
	}
}


void fresnelPropagator ( cufftComplex* frProp, params_t* params, params_t* params_d )
{
    // flag = 1 for forward propagation and -1 for backward propagation
    fresnelPropagatorDevice <<< params->CU.gS2D, params->CU.bS >>> ( frProp, params_d );
    zeroHighFreq <<< 2 * params->CU.gS2D, params->CU.bS >>> ( frProp, params->IM.m1, params->IM.m2 );
    //divide frProp with m1*m2, then the renormalization in the multislice later on does not need to be done explicitly anymore.
    const int m12 = params->IM.m1 * params->IM.m2;
    const float alpha = 1.f / ( ( float ) m12 );
    cublas_assert ( cublasCsscal ( params->CU.cublasHandle, m12, &alpha, ( cuComplex* ) frProp, 1 ) );
}


void convolveWithFrProp ( cufftComplex* psi, cufftComplex* frProp, params_t* params )
{
    cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, psi, psi, CUFFT_FORWARD ) );
    multiplyElementwise <<< params->CU.gS2D, params->CU.bS >>> ( psi, frProp, params->IM.m1 * params->IM.m2 );
    cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, psi, psi, CUFFT_INVERSE ) ); // No renormalization, frProp is scaled appropriately
}


void  applyLensFunction ( cufftComplex* psi, int k, params_t* params, params_t* params_d )
{
    const int m12 = params->IM.m1 * params->IM.m2;
    cufft_assert ( cufftExecC2C ( params->CU.cufftPlan,  psi, psi, CUFFT_FORWARD ) );
    multiplyLensFunction <<< params->CU.gS2D, params->CU.bS >>> ( psi, k, params_d );
    cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, psi, psi, CUFFT_INVERSE ) );
    const float alpha = 1.f / ( ( float ) m12 );
    cublas_assert ( cublasCsscal ( params->CU.cublasHandle, m12, &alpha, ( cuComplex* ) psi, 1 ) );
}

void saveTestImage ( char* fileName, cufftComplex* src, int m12 )
{
    cufftComplex* tomp;
    tomp = ( cufftComplex* ) malloc ( m12 * sizeof ( cufftComplex ) );
    cuda_assert ( cudaMemcpy ( tomp, src, m12 * sizeof ( cufftComplex ), cudaMemcpyDeviceToHost ) );

    float* tomp2;
    tomp2 = ( float* ) malloc ( m12 * sizeof ( float ) );

    char* fileName2;
    fileName2 = ( char* ) malloc ( 128 * sizeof ( char ) );

    for ( int el = 0; el <  m12; el++ )
    { tomp2[el] = tomp[el].x; }

    sprintf ( fileName2, "%s_re.bin", fileName );
    writeBinary ( fileName2, tomp2, m12 );

    for ( int el = 0; el <  m12; el++ )
    { tomp2[el] = tomp[el].y; }

    sprintf ( fileName2, "%s_im.bin", fileName );
    writeBinary ( fileName2, tomp2, m12 );

    for ( int el = 0; el <  m12; el++ )
    { tomp2[el] = tomp[el].x * tomp[el].x + tomp[el].y * tomp[el].y; }

    sprintf ( fileName2, "%s_I.bin", fileName );
    writeBinary ( fileName2, tomp2, m12 );

    free ( fileName2 );
    free ( tomp );
    free ( tomp2 );
}


void macroTest ( int m )
{
    int t;
    fprintf ( stderr, "\n    Test iwCoord and owCoord:\n    Test for m = %i\n    iwCoord: ", m );

    for ( int i = 0; i < ( m ); i++ )
    {
        iwCoord ( t, i, m );
        fprintf ( stderr, " %i", t );
    }

    fprintf ( stderr, "\n    owCoord: " );

    for ( int i = 0; i < m; i++ )
    {
        owCoord ( t, i, m );
        fprintf ( stderr, " %i", t );
    }

    m += 1;
    fprintf ( stderr, "\n    Test for m = %i\n    iwCoord: ", m );

    for ( int i = 0; i < m; i++ )
    {
        iwCoord ( t, i, m );
        fprintf ( stderr, " %i", t );
    }

    fprintf ( stderr, "\n    owCoord: " );

    for ( int i = 0; i < m; i++ )
    {
        owCoord ( t, i, m );
        fprintf ( stderr, " %i", t );
    }

    int m1 = 5;
    int m2 = 6;
    fprintf ( stderr, "\n\n    Test dbCoord and sgCoord:\n    Test for m1 = %i and m2 = %i\n", m1, m2 );

    for ( int i2 = 0; i2 < m2; i2++ )
    {
        for ( int i1 = 0; i1 < m1; i1++ )
        {
            int j;
            sgCoord ( j, i1, i2, m1 );
            int k1, k2;
            dbCoord ( k1, k2, j, m1 );
            fprintf ( stderr, "    (%i, %i) -> j = %i -> (%i, %i)\n", i1, i2, j, k1, k2 );
        }
    }

    fprintf ( stderr, "\n" );
}
