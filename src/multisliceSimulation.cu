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
#include "crystalMaker.h"


__global__ void potential2Transmission ( cufftComplex* V, int size )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    {
        float Vx = V[i].x;
        float Vy = V[i].y;
	V[i].x = expf ( -Vy ) * cosf ( Vx );
	V[i].y = expf ( -Vy ) * sinf ( Vx );
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

        if ( ( ( ( float ) ( i1 * i1 ) ) / ( ( float ) ( dim1 * dim1 ) ) + ( ( float ) ( i2 * i2 ) ) / ( ( float ) ( dim2 * dim2 ) ) ) < 0.25f )
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


__global__ void sphericalWavefront_d ( cufftComplex* psi, params_t* params )
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

		float r  = ( (float) i1 ) * params->IM.d1;
		float k = ( (float) i2 ) * params->IM.d2;
		float x3 = 1070e-9f; // From experiment
		r = sqrtf( r * r + k * k + x3 * x3 );
		k = 2.f * params->cst.pi * ( r / params->EM.lambda ) ; // recycle x1
		r = x3 / r;

		psi[i].x = r * cosf ( k );
		psi[i].y = r * sinf ( k );
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

	if ( i < dim1 * dim2 ){
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


__global__ void condensedBeam_d ( cufftComplex* psi, params_t* params )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int dim1 = params->IM.m1;
	const int dim2 = params->IM.m2;

	if ( i < dim1 * dim2 ) {
		int i1, i2;
		dbCoord ( i1, i2, i, dim1 );
		iwCoordIp ( i1, dim1 );
		iwCoordIp ( i2, dim2 );

		float th1 = ( (float) dim1 ) * params->IM.d1;
		float th2 = ( (float) dim2 ) * params->IM.d2;
		float Dz = 0.25f * sqrtf( th1 * th1 + th2 * th2 ) / tanf( params->EM.condensorAngle );

		th1 = ( ( (float) i1 ) / ( (float) dim1 ) ) * ( params->EM.lambda / params->IM.d1 );
		th2 = ( ( (float) i2 ) / ( (float) dim2 ) ) * ( params->EM.lambda / params->IM.d2 );
		th1 = th1 * th1 + th2 * th2;

		if( sqrtf( th1 ) < params->EM.condensorAngle ) {
			th2 = params->cst.pi * ( Dz / params->EM.lambda ) * th1;
			psi[i].x =  cosf( th2 ) / ( (float) ( dim1 * dim2 ) ); // Divide by dim1 * dim2 to account for the fftw artifact
			psi[i].y = -sinf( th2 ) / ( (float) ( dim1 * dim2 ) );
		}
		else {
			psi[i].x = 0.f;
			psi[i].y = 0.f;
		}
	}

}


__global__ void taperedCosineWindow_d ( cufftComplex* psi, params_t* params, float bckgrnd )
{
	// Apply the tapered cosine or Tukey window: http://en.wikipedia.org/wiki/Window_function

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int dim1 = params->IM.m1;
	const int dim2 = params->IM.m2;

	if ( i < dim1 * dim2 ) {
		int i1, i2;
		float w, alpha, x;
		dbCoord ( i1, i2, i, dim1 );

		w = 1.f;

		alpha = 1.5f * ( ( (float) params->IM.dn1 ) / ( (float) dim1 ) ); // alpha parameter of the tapered cosine window
		x = ( (float) i1 ) / ( (float) ( dim1 - 1) );
		if ( x < alpha * 0.5f )	{
			w = 0.5f * ( 1.f + cosf( params->cst.pi * ( 2.f * x / alpha - 1.f ) ) ); 
		} else { if ( x > 1.f - 0.5f * alpha ) {
			w = 0.5f * ( 1.f + cosf( params->cst.pi * ( 2.f * x / alpha + 1.f - 2.f / alpha ) ) );
		} }

		alpha = 1.5f * ( ( (float) params->IM.dn2 ) / ( (float) dim2 ) ); // alpha parameter of the tapered cosine window
		x = ( (float) i2 ) / ( (float) ( dim2 - 1) );
		if ( x < alpha * 0.5f )	{
			w *= 0.5f * ( 1.f + cosf( params->cst.pi * ( 2.f * x / alpha - 1.f ) ) ); 
		} else { if ( x > 1.f - 0.5f * alpha ) {
			w *= 0.5f * ( 1.f + cosf( params->cst.pi * ( 2.f * x / alpha + 1.f - 2.f / alpha ) ) ); 
		} }
		psi[i].x *= w;
		psi[i].x += ( 1.f - w ) * bckgrnd;
		psi[i].y *= w;
		psi[i].y += ( 1.f - w ) * bckgrnd;
	}
}

__global__ void zeroHighFreq ( cufftComplex* f, int dim1, int dim2 )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	float mindim = ( float ) dim1;
	if ( ( float ) dim2 < mindim ) {
		mindim = ( float ) dim2; 
	}
	if ( i < 2 * dim1 * dim2 ) {
		int i0 = i / 2;
		int i1, i2;
		dbCoord ( i1, i2, i0, dim1 );
		iwCoordIp ( i1, dim1 );
		iwCoordIp ( i2, dim2 );

		if ( ( ( (float) ( i1 * i1 + i2 * i2 ) ) * 9.f / ( mindim * mindim ) ) > 1.f ) {
			if ( ( i % 2 ) == 0 ) {
				f[i0].x = 0.f; 
			} else {
				f[i0].y = 0.f;
			}
        }
    }
}


__global__ void fresnelPropagatorDevice ( cufftComplex* frProp, params_t* params, int flag )
{
    // flag = 1 for forward propagation and -1 for backward propagation
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

    if ( i < dim1 * dim2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, i, dim1 );
        iwCoordIp ( i1, dim1 );
        iwCoordIp ( i2, dim2 );
        i1 *= flag;
        i2 *= flag;

        float d3 = params->IM.d3;
        const float t1 = ( ( (float) i1 ) / ( (float) dim1 ) ) * ( d3 / params->IM.d1 );
        const float t2 = ( ( (float) i2 ) / ( (float) dim2 ) ) * ( d3 / params->IM.d2 );
        d3 = params->EM.lambda / d3;
	d3 = -params->cst.pi * d3 * ( t1 * t1 + t2 * t2 );
        frProp[i].x = cosf ( d3 );
        frProp[i].y = sinf ( d3 );
    }
}


__global__ void shiftSpectiltBack ( cufftComplex* psi, int k, params_t* params, int flag )
{
	// NOTE: This is a remnant from when the rotation was approximate. Is not used anymore as of 22.02.016
    // flag == +1: shift the image back to the centre, for during the forward propagation
    // flag == -1: shift the image away from the centre, for during the backpropagation

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

    if ( i < dim1 * dim2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, i, dim1 );
        iwCoordIp ( i1, dim1 );
        iwCoordIp ( i2, dim2 );

        float m3p1 = ( float ) ( params->IM.m3 + 1 );
        float d3 = params->IM.d3;
		float k1p2 = params->cst.pi * ( ( float ) ( -flag ) ) *
                     ( ( ( float ) i1 ) * ( m3p1 / ( ( float ) dim1 ) ) * ( d3 / params->IM.d1 ) * tanf ( -params->IM.tiltspec[2 * k + 1] )
                     + ( ( float ) i2 ) * ( m3p1 / ( ( float ) dim2 ) ) * ( d3 / params->IM.d2 ) * tanf ( -params->IM.tiltspec[2 * k] ) );

		m3p1 = psi[i].x; // recycle m3p1
        d3   = psi[i].y; // recycle d3
        const float cosk1p2 = cosf ( k1p2 );
        const float sink1p2 = sinf ( k1p2 );

        psi[i].x = m3p1 * cosk1p2 - d3 * sink1p2; // real*real - imag*imag
        psi[i].y = m3p1 * sink1p2 + d3 * cosk1p2; // real*imag + imag*real
    }
}


__global__ void multiplyLensFunction ( cufftComplex* psi, int k, params_t* params, int flag )
{
    // flag = 1 for forwardPropagation and -1 for backwardPropagation

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

    if ( i < dim1 * dim2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, i, dim1 );
        iwCoordIp ( i1, dim1 );
        iwCoordIp ( i2, dim2 );
        i1 *= flag;
        i2 *= flag;

        float nu1  = ( ( ( float ) i1 ) / ( ( float ) dim1 ) ) * ( params->EM.lambda / params->IM.d1 );
        float nu2 = ( ( ( float ) i2 ) / ( ( float ) dim2 ) ) * ( params->EM.lambda / params->IM.d2 );
        float phi = atan2f ( nu2, nu1 );
        float nu = sqrtf ( nu1 * nu1 + nu2 * nu2 );

        if ( nu < params->EM.ObjAp )
        {
            float W = nu * nu * ( 0.5f * (
                                      params->EM.aberration.A1_0 * cosf ( 2.f * ( phi - params->EM.aberration.A1_1 ) )
									  + params->EM.aberration.C1_0 + params->IM.defoci[k]
									  - ( (float) ( params->IM.m3 + 1 ) ) * params->IM.d3 * 0.5f ) 
                                  + nu * ( 0.33333333f * (
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
                                                            + nu * ( 0.1665f * (
                                                                    params->EM.aberration.A5_0 * cosf ( 6.f * ( phi - params->EM.aberration.A5_1 ) )
                                                                    + params->EM.aberration.R5_0 * cosf ( 4.f * ( phi - params->EM.aberration.R5_1 ) )
                                                                    + params->EM.aberration.S5_0 * cosf ( 2.f * ( phi - params->EM.aberration.S5_1 ) )
                                                                    + params->EM.aberration.C5_0 )
                                                                   ) ) ) ) ); // CLOSE ALL THE BRACKETS!
			
			// Apply a phase ramp that undoes the real-space image shift due to the beam tilts
			/* float nuBeam = 0.f;
			nuBeam = sqrtf( params->IM.tiltbeam[2 * k] * params->IM.tiltbeam[2 * k] + params->IM.tiltbeam[2 * k + 1] * params->IM.tiltbeam[2 * k + 1] );
			  // The next portion is for undoing the image shift that comes about due to the combination of non-zero beam tilts and non-zero aberrations
			    // It appears that this shift is important in estimating the aberrations, so it's best not to undo it.
				// However, I've written the code and it seems a shame to delete it, so it stays in, but commented out.
				// When including it, don't forget to adjust multiplyDC1Prefactor_d etc. as well.
			if ( nuBeam < ( params->EM.ObjAp + FLT_EPSILON ) ) // i.e. only for bright field
			{
				W -= ( nu1 * params->IM.tiltbeam[2 * k + 1] + nu2 * params->IM.tiltbeam[2 * k] ) *
					( params->EM.aberration.C1_0 + params->RE.defocPoly0 + params->IM.defoci[k] * ( params->RE.defocPoly1 + params->IM.defoci[k] * params->RE.defocPoly2 )
					- ( (float) ( params->IM.m3 + 1 ) ) * params->IM.d3 * 0.5f
					+ params->EM.aberration.A1_0 * cosf ( 2.f * ( phi - params->EM.aberration.A1_1 ) ) 
					+ nuBeam * ( params->EM.aberration.A2_0 * cosf ( 3.f * ( phi - params->EM.aberration.A2_1 ) )
					            + params->EM.aberration.B2_0 * cosf ( phi - params->EM.aberration.B2_1 )
					            + nuBeam * ( params->EM.aberration.A3_0 * cosf ( 4.f * ( phi - params->EM.aberration.A3_1 ) )
					                       + params->EM.aberration.S3_0 * cosf ( 2.f * ( phi - params->EM.aberration.S3_1 ) )
							               + params->EM.aberration.C3_0     
						    	           + nuBeam * ( params->EM.aberration.A4_0 * cosf ( 5.f * ( phi - params->EM.aberration.A4_1 ) )
							                           + params->EM.aberration.B4_0 * cosf ( phi - params->EM.aberration.B4_1 )
							                           + params->EM.aberration.D4_0 * cosf ( 3.f * ( phi - params->EM.aberration.D4_1 ) ) 
								                       + nuBeam * ( params->EM.aberration.A5_0 * cosf ( 6.f * ( phi - params->EM.aberration.A5_1 ) )
									                              + params->EM.aberration.R5_0 * cosf ( 4.f * ( phi - params->EM.aberration.R5_1 ) )
									                              + params->EM.aberration.S5_0 * cosf ( 2.f * ( phi - params->EM.aberration.S5_1 ) )
									                              + params->EM.aberration.C5_0 )  )  )  )  );
			}*/

            // Recycle the variables:
			nu = sqrtf ( nu1 * nu1 + nu2 * nu2 );
            nu2 = params->EM.lambda;
            float damp = params->EM.defocspread * nu * nu / nu2;
            damp = expf ( -2.f * damp * damp );
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



__global__ void multiplyDC1Prefactor_d ( cufftComplex* psi, int k, params_t* params )
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
 
        float nu1  = ( ( ( float ) i1 ) / ( ( float ) dim1 ) ) * ( params->EM.lambda / params->IM.d1 );
        float nu2 = ( ( ( float ) i2 ) / ( ( float ) dim2 ) ) * ( params->EM.lambda / params->IM.d2 );
        
		// division with dim1*dim2 to prevent rescaling for the CUFFT in derivativeDefocus
		float preFactor = -( nu1 * nu1 + nu2 * nu2 );
		/* // Include here (and in the other multiplyDXX_XPrefactor_d-kernels) to account for the image-shift compensation explained in multiplyLensFunction.
		if( sqrtf( params->IM.tiltbeam[2 * k] * params->IM.tiltbeam[2 * k] + params->IM.tiltbeam[2 * k + 1 ] * params->IM.tiltbeam[2 * k + 1] ) < ( params->EM.ObjAp + FLT_EPSILON ) )
		{    preFactor += 2.f * ( params->IM.tiltbeam[2 * k + 1] * nu1 + params->IM.tiltbeam[2 * k] * nu2 ) ; }*/
		preFactor *= params->cst.pi / ( params->EM.lambda  * ( ( float ) ( dim1 * dim2 ) ) );

		nu1 = psi[i].x;
		psi[i].x = -psi[i].y * preFactor;
		psi[i].y = nu1 * preFactor;
    }
}


__global__ void multiplyDA1_0Prefactor_d ( cufftComplex* psi, int k, params_t* params )
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
 
        float nu1  = ( ( ( float ) i1 ) / ( ( float ) dim1 ) ) * ( params->EM.lambda / params->IM.d1 );
        float nu2 = ( ( ( float ) i2 ) / ( ( float ) dim2 ) ) * ( params->EM.lambda / params->IM.d2 );
        
		// division with dim1*dim2 to prevent rescaling for the CUFFT in derivativeDefocus
		float preFactor =  -( nu1 * nu1 + nu2 * nu2 );
		preFactor *= params->cst.pi / ( params->EM.lambda  * ( ( float ) ( dim1 * dim2 ) ) ) * cosf( 2.f * (  atan2f ( nu2, nu1 ) - params->EM.aberration.A1_1 ) );

		nu1 = psi[i].x;
		psi[i].x = -psi[i].y * preFactor;
		psi[i].y = nu1 * preFactor;
    }
}


__global__ void multiplyDA1_1Prefactor_d ( cufftComplex* psi, int k, params_t* params )
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
 
        float nu1  = ( ( ( float ) i1 ) / ( ( float ) dim1 ) ) * ( params->EM.lambda / params->IM.d1 );
        float nu2 = ( ( ( float ) i2 ) / ( ( float ) dim2 ) ) * ( params->EM.lambda / params->IM.d2 );
        
		// division with dim1*dim2 to prevent rescaling for the CUFFT in derivativeDefocus
		float preFactor = -( nu1 * nu1 + nu2 * nu2 );
		preFactor *=   params->cst.pi / ( params->EM.lambda  * ( ( float ) ( dim1 * dim2 ) ) ) 
			         * params->EM.aberration.A1_0 * sinf( 2.f * (  atan2f ( nu2, nu1 ) - params->EM.aberration.A1_1 ) ) * 2.f;

		nu1 = psi[i].x;
		psi[i].x = -psi[i].y * preFactor;
		psi[i].y = nu1 * preFactor;
    }
}


__global__ void multiplyDA2_0Prefactor_d ( cufftComplex* psi, int k, params_t* params )
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
 
        float nu1 = ( ( ( float ) i1 ) / ( ( float ) dim1 ) ) * ( params->EM.lambda / params->IM.d1 );
        float nu2 = ( ( ( float ) i2 ) / ( ( float ) dim2 ) ) * ( params->EM.lambda / params->IM.d2 );
        
		// division with dim1*dim2 to prevent rescaling for the CUFFT in derivativeDefocus

		float preFactor = -( nu1 * nu1 + nu2 * nu2 ) / 3.f;
		preFactor *= 2.f * params->cst.pi / ( params->EM.lambda  * ( ( float ) ( dim1 * dim2 ) ) )
			* sqrtf( nu1 * nu1 + nu2 * nu2 ) * cosf( 3.f * (  atan2f ( nu2, nu1 ) - params->EM.aberration.A2_1 ) );

		nu1 = psi[i].x;
		psi[i].x = -psi[i].y * preFactor;
		psi[i].y = nu1 * preFactor;
    }
}


__global__ void multiplyDA2_1Prefactor_d ( cufftComplex* psi, int k, params_t* params )
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
 
        float nu1  = ( ( ( float ) i1 ) / ( ( float ) dim1 ) ) * ( params->EM.lambda / params->IM.d1 );
        float nu2 = ( ( ( float ) i2 ) / ( ( float ) dim2 ) ) * ( params->EM.lambda / params->IM.d2 );
        
		// division with dim1*dim2 to prevent rescaling for the CUFFT in derivativeDefocus
		float preFactor = -( nu1 * nu1 + nu2 * nu2 );
		preFactor *= 2.f * params->cst.pi / ( params->EM.lambda  * ( ( float ) ( dim1 * dim2 ) ) )
			* sqrtf( nu1 * nu1 + nu2 * nu2 ) * params->EM.aberration.A2_0 * sinf( 3.f * (  atan2f ( nu2, nu1 ) - params->EM.aberration.A2_1 ) );

		nu1 = psi[i].x;
		psi[i].x = -psi[i].y * preFactor;
		psi[i].y = nu1 * preFactor;
    }
}


__global__ void multiplyDB2_0Prefactor_d ( cufftComplex* psi, int k, params_t* params )
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
 
        float nu1  = ( ( ( float ) i1 ) / ( ( float ) dim1 ) ) * ( params->EM.lambda / params->IM.d1 );
        float nu2 = ( ( ( float ) i2 ) / ( ( float ) dim2 ) ) * ( params->EM.lambda / params->IM.d2 );
        
		// division with dim1*dim2 to prevent rescaling for the CUFFT in derivativeDefocus
		float preFactor = -( nu1 * nu1 + nu2 * nu2 ) / 3.f;
		preFactor *= 2.f * params->cst.pi / ( params->EM.lambda  * ( ( float ) ( dim1 * dim2 ) ) )
			* sqrtf( nu1 * nu1 + nu2 * nu2 ) * cosf( atan2f ( nu2, nu1 ) - params->EM.aberration.B2_1 );

		nu1 = psi[i].x;
		psi[i].x = -psi[i].y * preFactor;
		psi[i].y = nu1 * preFactor;
    }
}


__global__ void multiplyDB2_1Prefactor_d ( cufftComplex* psi, int k, params_t* params )
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
 
        float nu1  = ( ( ( float ) i1 ) / ( ( float ) dim1 ) ) * ( params->EM.lambda / params->IM.d1 );
        float nu2 = ( ( ( float ) i2 ) / ( ( float ) dim2 ) ) * ( params->EM.lambda / params->IM.d2 );
        
		// division with dim1*dim2 to prevent rescaling for the CUFFT in derivativeDefocus
		float preFactor = -( nu1 * nu1 + nu2 * nu2 ) / 3.f;
		preFactor *= 2.f * params->cst.pi / ( params->EM.lambda  * ( ( float ) ( dim1 * dim2 ) ) )
			* sqrtf( nu1 * nu1 + nu2 * nu2 ) * params->EM.aberration.B2_0 * sinf( atan2f ( nu2, nu1 ) - params->EM.aberration.B2_1 );

		nu1 = psi[i].x;
		psi[i].x = -psi[i].y * preFactor;
		psi[i].y = nu1 * preFactor;
    }
}

__global__ void multiplyDC3Prefactor_d ( cufftComplex* psi, int k, params_t* params )
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
 
        float nu1  = ( ( ( float ) i1 ) / ( ( float ) dim1 ) ) * ( params->EM.lambda / params->IM.d1 );
        float nu2 = ( ( ( float ) i2 ) / ( ( float ) dim2 ) ) * ( params->EM.lambda / params->IM.d2 );
        
		// division with dim1*dim2 to prevent rescaling for the CUFFT in derivativeDefocus
		float preFactor =  -( nu1 * nu1 + nu2 * nu2 ) * 0.25f ;
		preFactor *= 2.f * params->cst.pi / ( params->EM.lambda  * ( ( float ) ( dim1 * dim2 ) ) ) * ( nu1 * nu1 + nu2 * nu2 );

		nu1 = psi[i].x;
		psi[i].x = -psi[i].y * preFactor;
		psi[i].y = nu1 * preFactor;
    }
}


__global__ void multiplySpatialFreqTilt_d ( cufftComplex* f, const int k, params_t* params, const int xy )
{
	// xy selects x- or y- tilt (by being 0 or 1 resp.)
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

    if ( i < dim1 * dim2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, i, dim1 );
        iwCoordIp ( i1, dim1 );
        iwCoordIp ( i2, dim2 ); // convention for forward prop, see fresnelPropagatorDevice

		float temp = cosf( params->IM.tiltspec[2*k + 1 - xy] );
		float fctr = 2.f * params->cst.pi / ( temp * temp );
		if ( xy == 0 )
		{    fctr *= ( ( float ) ( i1 ) / ( ( float ) dim1 ) ) * ( params->IM.d3 / params->IM.d1 ); }
		else // xy == 1
		{    fctr *= ( ( float ) ( i2 ) / ( ( float ) dim2 ) ) * ( params->IM.d3 / params->IM.d2 ); }

		temp = f[i].x;
		f[i].x = - fctr * f[i].y;
		f[i].y =   fctr * temp;
    }
}


__global__ void intensityValues ( cufftComplex* psi, int m12 )
{
    // Takes the values of one layer of the object and writes the intensity in the NEXT layer.

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < m12 )
    {
        float psiRe = psi[i].x;
        float psiIm = psi[i].y;
        psi[i + m12].x = psiRe * psiRe + psiIm * psiIm;
        psi[i + m12].y = 0.f;
    }
}

__global__ void intensityValsInplc_d ( cufftComplex* psi, int m12 )
{
	// Takes the values of one layer of the object and writes the intensity in the SAME layer.

	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i < m12 ) {
		float psiRe = psi[i].x;
		float psiIm = psi[i].y;
		psi[i].x = psiRe * psiRe + psiIm * psiIm;
		psi[i].y = 0.f;
	}
}


__global__ void multiplyMtf ( cufftComplex* psi, params_t* params, int flag )
{
	// flag = 1 for forwardPropagation and -1 for backwardPropagation

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int dim1 = params->IM.m1;
	const int dim2 = params->IM.m2;

	if ( i < dim1 * dim2 ) {
		int i1, i2;
		dbCoord ( i1, i2, i, dim1 );
		iwCoordIp ( i1, dim1 );
		iwCoordIp ( i2, dim2 );
		i1 *= flag;
		i2 *= flag;

		float nu1 = ( ( float ) i1 ) / ( ( float ) dim1 );
		float nu2 = ( ( float ) i2 ) / ( ( float ) dim2 );
		float mtf = sqrtf( nu1 * nu1 + nu2 * nu2 );

		mtf = ( params->EM.mtfa * expf ( -params->EM.mtfc * mtf ) + params->EM.mtfb * expf ( -params->EM.mtfd * mtf * mtf ) );

		nu1 *= params->cst.pi;
		nu2 *= params->cst.pi;
		// mtf *= ( ( sinf ( nu1 ) + FLT_EPSILON ) / ( nu1 + FLT_EPSILON ) ) * ( ( sinf ( nu2 ) + FLT_EPSILON ) / ( nu2 + FLT_EPSILON ) );
		if ( fabsf( nu1 ) > 1e-3f ) {
			mtf *= sinf ( nu1 ) / nu1;
		}
		if ( fabsf( nu2 ) > 1e-3f ) {
			mtf *= sinf ( nu2 ) / nu2;
		}

		psi[i].x *= mtf;
		psi[i].y *= mtf;
	}
}


__global__ void multiplySpatialIncoherence ( cufftComplex* psi, int k, params_t* params, int flag )
{
    // flag = 1 for forwardPropagation and -1 for backwardPropagation
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim1 = params->IM.m1;
    const int dim2 = params->IM.m2;

    if ( i < dim1 * dim2 )
    {
        int i1, i2;
        dbCoord ( i1, i2, i, dim1 );
        iwCoordIp ( i1, dim1 );
        iwCoordIp ( i2, dim2 );
        i1 *= flag;
        i2 *= flag;

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


__global__ void zeroImagPart ( cufftComplex* f, int size )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    { f[i].y = 0.f; }
}


__global__ void multiplyComplexConjugate ( cufftComplex* f0, cufftComplex* f1, int m12 )
{
    // multiplies THE NEXT LAYER of f0 with the complex conjugate of f1 and puts it in the CURRENT layer of f0 (because its BACKpropagation)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < m12 )
    {
        float f0Re = f0[i + m12].x;
        float f0Im = f0[i + m12].y;
        float f1Re = f1[i].x;
        float f1Im = -f1[i].y;
        float temp = f0Re * ( f1Re + f1Im );
        f1Im = f1Im * ( f0Re + f0Im );
        f1Re = f1Re * ( f0Im - f0Re );
        f0[i].x = temp - f1Im;
        f0[i].y = temp + f1Re;
    }
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


__global__ void flattenDynamicRange( cuComplex* f, int size, int flag )
{
	// flag == 1: flatten, flag == -1: unflatten

	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	float signFx = 1.f;
	float signFy = 1.f;
	
	if ( j < size )
	{
		if( f[j].x < 0.f )
		{    signFx = -1.f; }
		if( f[j].y < 0.f )
		{    signFy = -1.f; }

		// Flatten the dynamic range so that linear interpolation can work better
		if ( flag == 1 )
		{
			f[j].x = signFx * log10f( fabsf( f[j].x ) + 1.f );
			f[j].y = signFy * log10f( fabsf( f[j].y ) + 1.f );
		}
		// Unflatten the dynamic range
		if ( flag == -1 )
		{
			f[j].x = signFx * ( powf( 10.f, f[j].x * signFx ) - 1.f );
			f[j].y = signFy * ( powf( 10.f, f[j].y * signFy ) - 1.f );
		}
	}
}

__global__ void tiltObject_d ( cuComplex* fOut, cuComplex* fIn, int it, params_t* params, int flag )
{
	// flag == 1: forward rotation, flag == -1: rotate back
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    
	const int m1 = params->IM.m1;
	const int m2 = params->IM.m2;
	const int m3 = params->IM.m3;

	float s = 0.f;
	float c = 1.f;

	int i1, i2, i3, k;
	float x1, x2, x3, xTemp;

	float fOutx = 0.f;
	float fOuty = 0.f;

    if ( j < ( m1 * m2 * m3 ) )
    {
		trCoord( i1, i2, i3, j, m1, m2 );
		x1 = ( (float) i1 ) - ( (float) m1 ) * 0.5f;
		x2 = ( ( (float) i2 ) - ( (float) m2 ) * 0.5f ) * ( params->IM.d2 / params->IM.d1 );
		x3 = ( ( (float) i3 ) - ( (float) m3 ) * 0.5f ) * ( params->IM.d3 / params->IM.d1 );

		// Rotation. This is compatible with FDES rotations. 
		// The angle has a differnt sign than FDES, because in FDES we rotate the OBJECT, 
		// while in the following lines we rotate the AXES
		if ( flag == 1 )
		{
			// First rotation about the second dimension (x-axis) (this is tilt_y, SORRY FOR THE MISNOMER)
			c =  cosf( params->IM.tiltspec[2*it+1] );
			s = -sinf( params->IM.tiltspec[2*it+1] );
			xTemp = c * x1 - s * x3;
			x3   *= c;
			x3   += s * x1;
			x1 = xTemp;
			// Then rotation about the first dimension (y-axis) (this is tilt_x, SORRY FOR THE MISNOMER)
			c =  cosf( params->IM.tiltspec[2*it] );
			s = -sinf( params->IM.tiltspec[2*it] );
			xTemp = c * x2 - s * x3;
			x3   *= c;
			x3   += s * x2;
			x2 = xTemp;
		}

		if ( flag == -1 )
		{
			// First rotation about the first dimension (y-axis) (this is tilt_x, SORRY FOR THE MISNOMER)
			c = cosf( params->IM.tiltspec[2*it] );
			s = sinf( params->IM.tiltspec[2*it] );
			xTemp = c * x2 - s * x3;
			x3   *= c;
			x3   += s * x2;
			x2 = xTemp;
			// Then rotation about the second dimension (x-axis) (this is tilt_y, SORRY FOR THE MISNOMER)
			c = cosf( params->IM.tiltspec[2*it+1] );
			s = sinf( params->IM.tiltspec[2*it+1] );
			xTemp = c * x1 - s * x3;
			x3   *= c;
			x3   += s * x1;
			x1 = xTemp;
		}

		// Then trilinear interpolation, following http://paulbourke.net/miscellaneous/interpolation/
		// Rescale the variables to have vox size of 1 and have their origin where the arrays start
		x1 += ( (float) m1 ) * 0.5f; // Again, the d1 = 1 and thus implicit
		x2 *= 1.f / ( params->IM.d2 / params->IM.d1 );
		x2 += ( (float) m2 ) * 0.5f;
		x3 *= 1.f / ( params->IM.d3 / params->IM.d1 );
		x3 += ( (float) m3 ) * 0.5f;

		i1 = roundf( x1 - 0.5f );
		i2 = roundf( x2 - 0.5f );
		i3 = roundf( x3 - 0.5f );
		
		x1 += -( (float) i1 );
		x2 += -( (float) i2 );
		x3 += -( (float) i3 );

		if( ( i1 > -1 ) && ( i1 < ( m1 - 1 ) ) && ( i2 > -1 ) && ( i2 < ( m2 - 1 ) ) && ( i3 > -1 ) && ( i3 < ( m3 - 1 ) ) )
		{
			sgCoord3D( k, i1, i2, i3, m1, m2 );
			fOutx  = fIn[k].x * ( 1.f - x1 ) * ( 1.f - x2 ) * ( 1.f - x3 );
			fOuty  = fIn[k].y * ( 1.f - x1 ) * ( 1.f - x2 ) * ( 1.f - x3 );

			sgCoord3D( k, i1 + 1, i2, i3, m1, m2 );
			fOutx += fIn[k].x * x1 * ( 1.f - x2 ) * ( 1.f - x3 );
			fOuty += fIn[k].y * x1 * ( 1.f - x2 ) * ( 1.f - x3 );

			sgCoord3D( k, i1, i2 + 1, i3, m1, m2 );
			fOutx += fIn[k].x * ( 1.f - x1 ) * x2 * ( 1.f - x3 );
			fOuty += fIn[k].y * ( 1.f - x1 ) * x2 * ( 1.f - x3 );

			sgCoord3D( k, i1, i2, i3 + 1, m1, m2 );
			fOutx += fIn[k].x * ( 1.f - x1 ) * ( 1.f - x2 ) * x3;
			fOuty += fIn[k].y * ( 1.f - x1 ) * ( 1.f - x2 ) * x3;

			sgCoord3D( k, i1 + 1, i2 + 1, i3, m1, m2 );
			fOutx += fIn[k].x * x1 * x2 * ( 1.f - x3 );
			fOuty += fIn[k].y * x1 * x2 * ( 1.f - x3 );

			sgCoord3D( k, i1, i2 + 1, i3 + 1, m1, m2 );
			fOutx += fIn[k].x * ( 1.f - x1 ) * x2 * x3;
			fOuty += fIn[k].y * ( 1.f - x1 ) * x2 * x3;

			sgCoord3D( k, i1 + 1, i2, i3 + 1, m1, m2 );
			fOutx += fIn[k].x * x1 * ( 1.f - x2 ) * x3;
			fOuty += fIn[k].y * x1 * ( 1.f - x2 ) * x3;

			sgCoord3D( k, i1 + 1, i2 + 1, i3 + 1, m1, m2 );
			fOutx += fIn[k].x * x1 * x2 * x3;
			fOuty += fIn[k].y * x1 * x2 * x3;
		}
		fOut[j].x = fOutx;
		fOut[j].y = fOuty;
    }
}


void rawMeasurement( cufftComplex *I_d, params_t* params, params_t* params_d, int k, int nAt, int nZ, float *xyzCoord_d, 
		    float imPot, int *Z_d, int *Zlist, float *DWF_d, float *occ_d, curandState *dwfState_d, int* itCnt )
{
	float *xyzCoordFP_d;
	cufftComplex *V_d, *frProp_d;
	int m12 = params->IM.m1 * params->IM.m2;

	cuda_assert ( cudaMalloc ( ( void** ) &xyzCoordFP_d, nAt * 3 * sizeof ( float ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &V_d, m12 * sizeof ( cufftComplex ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &frProp_d, m12 * sizeof ( cufftComplex ) ) );

	fresnelPropagator ( frProp_d, params, params_d, 1 );
	cublas_assert ( cublasScopy ( params->CU.cublasHandle, nAt * 3, xyzCoord_d, 1, xyzCoordFP_d, 1 ) );
	if ( params->IM.frPh > 0) {   
		atomJitter( xyzCoordFP_d, dwfState_d, nAt, DWF_d ); 
	}
	float temp = 0.f;
	incomingWave( I_d, params, params_d, k );
	for ( int i = 0; i < params->IM.m3; i++ ) {
		(*itCnt) = progressCounter ( *itCnt, params );
		phaseGrating( V_d, nAt, nZ, params, params_d, xyzCoordFP_d, imPot, Z_d, Zlist, occ_d, i );
		forwardPropagation( I_d, V_d, frProp_d, params, params_d );
	}
	if ( params->IM.mode == 0 ) {
		applyLensFunction ( I_d, k, params, params_d ); // Apply the lens function
	}
	if ( ( params->IM.mode == 1 ) || ( params->IM.mode == 2 ) || ( params->IM.mode == 3 ) ) {
		farFieldTransform( I_d, params, params_d );
	}
	intensityValsInplc_d <<< params->CU.gS2D, params->CU.bS >>> ( I_d, m12 ); // Take the intensity
	cuda_assert ( cudaDeviceSynchronize () );

	cuda_assert ( cudaFree ( xyzCoordFP_d ) );
	cuda_assert ( cudaFree ( V_d ) );
	cuda_assert ( cudaFree ( frProp_d ) );
}

void forwardPropagation ( cufftComplex* psi, cufftComplex* t, cufftComplex* frProp, params_t* params, params_t* params_d )
{
	const int m12 = params->IM.m1 * params->IM.m2;
	const int gS = params->CU.gS2D;
	const int bS = params->CU.bS;
	
	// Do the propagation 
	potential2Transmission <<< gS, bS>>> ( t, m12 ); // transform potential to transmission
	cuda_assert ( cudaDeviceSynchronize () );
	bandwidthLimit ( t, params );
	multiplyElementwise <<< gS, bS>>> ( psi, t, m12 ); // do the transmission
	cuda_assert ( cudaDeviceSynchronize () );
	convolveWithFrProp ( psi, frProp, params ); // do the propagation	
}

void bandwidthLimit ( cufftComplex* f, params_t* params )
{
	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, f, f, CUFFT_FORWARD ) );
	cuda_assert ( cudaDeviceSynchronize () );
	zeroHighFreq <<< 2 * params->CU.gS2D, params->CU.bS >>> ( f, params->IM.m1, params->IM.m2 );
	cuda_assert ( cudaDeviceSynchronize () );
	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, f, f, CUFFT_INVERSE ) );
	cuda_assert ( cudaDeviceSynchronize () );
	const int m12 = params->IM.m1 * params->IM.m2;
	const float alpha = 1.f / ( ( float ) m12 );
	cublas_assert ( cublasCsscal ( params->CU.cublasHandle, m12, &alpha, ( cuComplex* ) f, 1 ) );
	cuda_assert ( cudaDeviceSynchronize () );
}

void incomingWave( cufftComplex* psi, params_t* params, params_t* params_d, int k )
{
	const int gS = params->CU.gS2D;
	const int bS = params->CU.bS;
	const int m12 = params->IM.m1 * params->IM.m2;
	float alpha = 1.f;

	// Plane wave for mode == 0 or 1 
	initialValues <<< 2 * gS, bS >>> ( psi, m12, 1.f, 0.f );

	// In case of structured illumination:
	// readIncomingWave ( psi, k % 8, params );

	if ( ( params->IM.mode == 2 ) || ( params->IM.mode == 3 ) ) {
		multiplyLensFunction <<< gS, bS >>> ( psi, k, params_d, 1 );
		cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, psi, psi, CUFFT_INVERSE ) );
		cufftShift2D_h( psi, params->IM.m1, params->IM.m2, bS );
		bandwidthLimit ( psi, params );
		cublas_assert( cublasScnrm2( params->CU.cublasHandle, m12, psi, 1, &alpha ) );
		if ( params->IM.mode == 2 ) {		
			alpha = sqrtf( (float) m12 ) / alpha;
		} else { // i.e. when mode == 3
			alpha = 1.f / alpha;
		}
		cublas_assert( cublasCsscal( params->CU.cublasHandle, m12, &alpha, psi, 1 ) );
	}
	if ( params->IM.doBeamTilt ) {
		tiltBeam_d <<< params->CU.gS, params->CU.bS >>> ( psi, k, params_d, 1 ); 
	}
}


void fresnelPropagator ( cufftComplex* frProp, params_t* params, params_t* params_d, int flag )
{
	// flag = 1 for forward propagation and -1 for backward propagation
	fresnelPropagatorDevice <<< params->CU.gS2D, params->CU.bS >>> ( frProp, params_d, flag );
	cuda_assert ( cudaDeviceSynchronize () );
	zeroHighFreq <<< 2 * params->CU.gS2D, params->CU.bS >>> ( frProp, params->IM.m1, params->IM.m2 );
	cuda_assert ( cudaDeviceSynchronize () );
	//divide frProp with m1*m2, then the renormalization in the multislice later on does not need to be done explicitly anymore.
	const int m12 = params->IM.m1 * params->IM.m2;
	const float alpha = 1.f / ( ( float ) m12 );
	cublas_assert ( cublasCsscal ( params->CU.cublasHandle, m12, &alpha, ( cuComplex* ) frProp, 1 ) );
	cuda_assert ( cudaDeviceSynchronize () );
}


void convolveWithFrProp ( cufftComplex* psi, cufftComplex* frProp, params_t* params )
{
	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, psi, psi, CUFFT_FORWARD ) );
	cuda_assert ( cudaDeviceSynchronize () );
	multiplyElementwise <<< params->CU.gS2D, params->CU.bS >>> ( psi, frProp, params->IM.m1 * params->IM.m2 );
	cuda_assert ( cudaDeviceSynchronize () );
	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, psi, psi, CUFFT_INVERSE ) ); // No renormalization, frProp is scaled appropriately
	cuda_assert ( cudaDeviceSynchronize () );
}


void applyLensFunction ( cufftComplex* psi, int k, params_t* params, params_t* params_d )
{
	const int m12 = params->IM.m1 * params->IM.m2;   
	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, psi, psi, CUFFT_FORWARD ) );
	cuda_assert ( cudaDeviceSynchronize () );
	// NOTE: From when the rotation was approximate: shiftSpectiltBack <<< params->CU.gS2D, params->CU.bS >>> ( & ( psi[m123 + m12] ), k, params_d, 1 );
	// cuda_assert ( cudaDeviceSynchronize () );
	// IS A SHIFTBEAMTILTBACK NECESSARY? SO FAR I THINK NOT, BUT KEEP IT IN MIND JUST IN CASE
	multiplyLensFunction <<< params->CU.gS2D, params->CU.bS >>> ( psi, k, params_d, 1 );
	cuda_assert ( cudaDeviceSynchronize () );
	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, psi, psi, CUFFT_INVERSE ) );
	cuda_assert ( cudaDeviceSynchronize () );
	const float alpha = 1.f / ( ( float ) m12 );
	cublas_assert ( cublasCsscal ( params->CU.cublasHandle, m12, &alpha, ( cuComplex* ) psi, 1 ) );
	cuda_assert ( cudaDeviceSynchronize () );
}


void applyMtfAndIncoherence ( cufftComplex* psi, int k, params_t* params, params_t* params_d, int flag )
{       // set flag to 1 to convolve a STEM scan (ie mode == 3) with an MTF, it's then the incoherent sources size
	const int m12 = params->IM.m1 * params->IM.m2;

	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, psi, psi, CUFFT_FORWARD ) );
	cuda_assert ( cudaDeviceSynchronize () );

	if ( ( params->IM.mode == 0 ) || ( params->IM.mode == 1 ) 
		|| ( params->IM.mode == 2 ) || ( flag == 1 ) ){
		multiplyMtf <<< params->CU.gS2D, params->CU.bS >>> ( psi, params_d, 1 );
		cuda_assert ( cudaDeviceSynchronize () );
	}
	if ( params->IM.mode == 0 ) {
		multiplySpatialIncoherence <<< params->CU.gS2D, params->CU.bS >>> ( psi, k, params_d, 1 );
		cuda_assert ( cudaDeviceSynchronize () );
	}

	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, psi, psi, CUFFT_INVERSE ) );
	cuda_assert ( cudaDeviceSynchronize () );

	const float alpha = 1.f / ( ( float ) m12 );
	cublas_assert ( cublasCsscal ( params->CU.cublasHandle, m12, &alpha, ( cuComplex* ) psi, 1 ) );
	cuda_assert ( cudaDeviceSynchronize () );
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
