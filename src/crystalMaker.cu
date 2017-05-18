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

#include "crystalMaker.h"


__global__ void setupCurandState_d( curandState *state, int seed, int size )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Each thread gets same seed, a different sequence number, no offset
    if ( i < size )
	{    curand_init(seed, i, 0, &state[i]); }
}

__global__ void atomJitter_d( float* xyz_d, float* dwf_d, int nAt, curandState *state )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < 3 * nAt )
	{
		curandState localState = state[i]; //Copy state to local memory for efficiency
		float x = curand_normal ( &localState );
		xyz_d[i] += x * 0.112539540f * sqrtf( dwf_d[i/3] ); // 0.11 = 1/(pi*sqrt(8))
		state[i] = localState;  // Copy (modified) state back to global memory
	}
}

__global__ void ascombeNoise_d( cufftComplex* f, float dose, int size, curandState *state ) // equation ??
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
	{
		curandState localState = state[i]; //Copy state to local memory for efficiency
		float fi = f[i].x * dose;
		if ( fi > 1e-2f )
		{
			float x = curand_normal ( &localState );
			x *= sqrtf( 1 - expf( -fi / 0.777134f ) );
			x += 2.f * sqrtf( fi + 0.375f ) - 0.25f / sqrtf( fi );
			x = roundf( 0.25f * x * x  - 0.375f );
			if ( x < FLT_MIN )
			{    x = 0.f; }
			f[i].x = x / dose;
		}
		state[i] = localState;  // Copy (modified) state back to global memory
	}
}


__global__ void squareAtoms_d ( cufftComplex* V, params_t* params, int nAt, int* Z, int Z0, float* xyz, float imPot, float* occ, int s )//double linear interpolation
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < nAt )
	{
		if ( Z[i] == Z0 )
		{
			const int m1 = params->IM.m1;
			const int m2 = params->IM.m2;
			const int m3 = params->IM.m3;
			float x1, x2;
			x1 = xyz[i * 3 + 0] / params->IM.d1 + ( (float) m1 ) * 0.5f - 0.5f;
			x2 = xyz[i * 3 + 1] / params->IM.d2 + ( (float) m2 ) * 0.5f - 0.5f;
			int i3 = (int) ( roundf( xyz[i * 3 + 2] / params->IM.d3 + ( (float) m3 ) * 0.5f - 0.5f ) );

			if ( ( ( x1 > 1.f ) && ( x1 < ( (float) ( m1 - 2 ) ) ) )  &&  ( ( x2 > 1.f ) && ( x2 < ( (float) ( m2 - 2 ) ) ) )  &&  ( ( i3 > s-1 ) && ( i3 <= s ) ) )
			{
				int i1 = (int) roundf( x1 );
				int i2 = (int) roundf( x2 );
				int j;
				float r1 = x1 - ( (float) i1 );
				float r2 = x2 - ( (float) i2 );
				float temp;

				sgCoord(j, i1, i2, m1);
				temp = ( 1 - fabsf( r1 ) ) * ( 1 - fabsf( r2 ) ) * occ[i];
				atomicAdd( &( V[j].x ), temp );
				atomicAdd( &( V[j].y ), temp * imPot );

				i2 += mySignum_d( r2 );
				sgCoord(j, i1, i2, m1);
				temp = ( 1 - fabsf( r1 ) ) * fabsf( r2 ) * occ[i];
				atomicAdd( &( V[j].x ), temp );
				atomicAdd( &( V[j].y ), temp * imPot );

				i1 += mySignum_d( r1 );
				sgCoord(j, i1, i2, m1);
				temp = fabsf( r1 ) * fabsf( r2 ) * occ[i];
				atomicAdd( &( V[j].x ), temp );
				atomicAdd( &( V[j].y ), temp * imPot );

				i2 -= mySignum_d( r2 );
				sgCoord(j, i1, i2, m1);
				temp = fabsf( r1 ) * ( 1 - fabsf( r2 ) ) * occ[i];
				atomicAdd( &( V[j].x ), temp );
				atomicAdd( &( V[j].y ), temp * imPot );
			}
		}
	}
}

__device__ int mySignum_d( float x )
{
	int i;
	if ( x < 0.f )
	{    i = -1; }
	else
	{    i =  1; }

	return( i );
}

__global__ void divideBySinc ( cufftComplex* V, params_t* params )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int m1 = params->IM.m1;
	const int m2 = params->IM.m2;

    if ( i <  m1 * m2 )
	{
		int i1, i2;
		dbCoord ( i1, i2, i, m1 );
        iwCoordIp ( i1, m1 );
        iwCoordIp ( i2, m2 );

		float y = params->cst.pi;
		float x = ( (float) i1 ) / ( (float) m1 ) * y;
		x  = ( x + FLT_EPSILON ) / ( sinf( x ) + FLT_EPSILON );
		y *= ( (float) i2 ) / ( (float) m2 );
		x *= ( y + FLT_EPSILON ) / ( sinf( y ) + FLT_EPSILON );

		V[i].x *= x;
		V[i].y *= x;
	}
}

__global__ void multiplyWithProjectedPotential_d ( cufftComplex* V1, cufftComplex* V2, params_t* params )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int m1 = params->IM.m1;
	const int m2 = params->IM.m2;

	if ( i < m1 * m2  )
	{
		float V2x = V2[i].x;
		V1[i].x *= V2x;
		V1[i].y *= V2x;
	}
}

__global__ void memorySetZero_d ( cufftComplex* V, params_t* params )
{
  	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int m1 = params->IM.m1;
	const int m2 = params->IM.m2;

	if ( i < m1 * m2  )
	{
		V[i].x =0;
		V[i].y =0;
	}
}

__global__ void areaWeighting ( cufftComplex* psi, cufftComplex* psi_gaussian, float* mask, params_t* params )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int m1 = params->IM.m1;
	const int m2 = params->IM.m2;

	if ( i < m1 * m2  )
	{
		float real, imag;
		real = psi_gaussian[i].x * ( 1 - mask[i] ) + psi[i].x * mask[i];
		imag = psi_gaussian[i].y * ( 1 - mask[i] ) + psi[i].y * mask[i];

		psi[i].x = real;
		psi[i].y = imag;
	}
}


void applyMaskFiltering( cufftComplex* psi, params_t* params, params_t* params_d )
{
      const int m12 = params->IM.m1 * params->IM.m2;

      float * mask;
      cufftComplex * gaussian, *psi_gaussian;
      cuda_assert ( cudaMalloc ( ( void** ) &mask,     m12 * sizeof ( cufftComplex ) ) );
      cuda_assert ( cudaMalloc ( ( void** ) &gaussian, m12 * sizeof ( cufftComplex ) ) );
      cuda_assert ( cudaMalloc ( ( void** ) &psi_gaussian, m12 * sizeof ( cufftComplex ) ) );

      areaMask <<< params->CU.gS, params->CU.bS >>> ( mask,  params_d );

	  initialValues <<< params->CU.gS * 2, params->CU.bS >>> ( gaussian, m12, 1.f, 0.f );

	  areaWeighting <<< params->CU.gS, params->CU.bS>>>  ( psi, gaussian, mask, params_d);

      cuda_assert ( cudaFree ( mask ) );
      cuda_assert ( cudaFree ( gaussian ) );
      cuda_assert ( cudaFree ( psi_gaussian ) );
}


void buildMeasurements( params_t* params, int* Z_d, float* xyzCoord_d, float* DWF_d, float* occ_d, char * image_name, char * emd_name)
{


  	int nAt = (*params).SAMPLE.nAt;
	int frPh = (*params).IM.frPh;
	float pD = (*params).IM.pD;
	float imPot =(*params).SAMPLE.imPot;
	float subSlTh =(*params).IM.subSlTh;

	float *xyzCoordTO_d, *xyzCoordFP_d, *J_d;
	cufftComplex *V_d, *I_d, *psi, *exitwave_d, *t, *frProp, *GP;

	int count, n123, m12, m123, nZ, *Zlist, k;
	curandState *dwfState_d, *poissonState_d;
	params_t *params_d;
	cufftHandle cufftPlanBatch;

	setDeviceParams ( &params_d, params );
	subSlTh = subSliceRatio( params->IM.d3, subSlTh );
	setSubSlices( params, params_d, subSlTh );

	count = 1;
	n123 = params->IM.n1 * params->IM.n2 * params->IM.n3;
	m12 = params->IM.m1 * params->IM.m2;
	m123 = params->IM.m1 * params->IM.m2 *params->IM.m3;
	k = 0;

	cuda_assert ( cudaMalloc ( ( void** ) &xyzCoordTO_d, nAt * 3 * sizeof ( float ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &xyzCoordFP_d, nAt * 3 * sizeof ( float ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &V_d, m12 * sizeof ( cufftComplex ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &I_d, m12 * sizeof ( cufftComplex ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &psi, m12 * sizeof ( cufftComplex ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &t,   m12 * sizeof ( cufftComplex ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &frProp, m12 * sizeof ( cufftComplex ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &GP,  m12 * sizeof ( cufftComplex ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &J_d, n123 * sizeof ( float ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &dwfState_d, 3 * nAt * sizeof( curandState ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &poissonState_d, m12 * sizeof( curandState ) ) );

	if(printLevel >1)
	{
	  cuda_assert ( cudaMalloc ( ( void** ) &exitwave_d, m12 * sizeof ( cufftComplex ) ) );
	  exitwave = ( float* ) malloc (2* params->IM.n3* m12*  sizeof ( float ) );

	}

 	image = ( float* ) malloc ( n123*  sizeof ( float ) );
	Zlist = ( int* ) malloc ( 103 * sizeof ( int ) );

	float tOff[3];
 	tOff[0] = (*params).IM.specimen_tilt_offset_x;
	tOff[1] = (*params).IM.specimen_tilt_offset_y;
	tOff[2] = (*params).IM.specimen_tilt_offset_z;

	cublas_assert ( cublasScopy ( params->CU.cublasHandle, nAt * 3, xyzCoord_d, 1, xyzCoordTO_d, 1 ) );
	tiltCoordinates( xyzCoordTO_d, nAt, tOff[0], tOff[1], tOff[2], params ); // Create coordinates with the tilt offset.

	nZ = listOfElements( Zlist, nAt, Z_d );

	fprintf (stderr, " Number of elements in the specimen: %d \n ",nZ );

	setCufftPlanBatch( &cufftPlanBatch, params );

	if ( frPh > 0 )
	{    setupCurandState_d <<< myGSize( 3 * nAt ), myBSize( 3 * nAt ) >>> ( dwfState_d, 1, 3 * nAt ); }

	if ( pD > FLT_MIN )
	{    setupCurandState_d <<< myGSize( m12 ), myBSize( m12 ) >>> ( poissonState_d, 1 + params->IM.n3, m12 ); }

	if ( frPh > 0 )
	{   count = frPh; }

	int j,s;

 	cufftComplex alpha;
	alpha.x = 1.f / ( (float) count );
	alpha.y = 0.f;

	char IM_mode[BUZZ_SIZE];
	switch(params->IM.mode)
	{
	case 0:
		sprintf(IM_mode ,"TEM Imaging");
		break;
	case 1:
		sprintf(IM_mode ,"DP");
		break;
	case 2:
		sprintf(IM_mode ,"CBED");
		break;
	default:
		exit(0);
	}

	fprintf (stderr, " Imaging mode: %s \n ",IM_mode);

	for ( k = 0; k < params->IM.n3; k++ )
	{
		initialValues <<< params->CU.gS2D * 2, params->CU.bS >>> ( I_d,  m12, 0.f, 0.f );
		if(printLevel >1)
		{    initialValues <<< params->CU.gS2D * 2, params->CU.bS >>> ( exitwave_d,  m12, 0.f, 0.f ); }

		cublas_assert ( cublasScopy ( params->CU.cublasHandle, nAt * 3, xyzCoordTO_d, 1, xyzCoord_d, 1 ) );
		tiltCoordinates( xyzCoord_d, nAt, params->IM.tiltspec[2 * k], params->IM.tiltspec[2 * k + 1], 0.f, params );
		for (  j = 0; j < count; j++ )
		{
			incomingWave( psi, k, params, params_d );
			cublas_assert ( cublasScopy ( params->CU.cublasHandle, nAt * 3, xyzCoord_d, 1, xyzCoordFP_d, 1 ) );
			if ( frPh > 0)
			{   atomJitter( xyzCoordFP_d, dwfState_d, nAt, DWF_d ); }

			for (s = 0; s < params->IM.m3; s++ )
			{
				progressCounter ( k * params->IM.m3 * count + j * params->IM.m3 + s , params->IM.n3 * count * params->IM.m3 );
				phaseGrating( V_d, nAt, nZ, params, params_d, xyzCoordFP_d, imPot, Z_d, Zlist, occ_d, cufftPlanBatch, s);
				forwardPropagation ( psi, V_d, frProp, t, params, params_d );
			}

			if(printLevel >1)
			{    cublas_assert( cublasCaxpy ( params->CU.cublasHandle, m12, &alpha, psi, 1, exitwave_d, 1) ); }

			if( params->IM.mode == 0 )
			{
				applyLensFunction ( psi, k, params, params_d );
				intensityValues <<< params->CU.gS, params->CU.bS>>> ( psi, m12 );
				cublas_assert( cublasCaxpy ( params->CU.cublasHandle, m12, &alpha, psi, 1, I_d, 1) );
			}

			else if( params->IM.mode == 1 )
			{
				diffractionPattern( psi, k, params, params_d );
				cublas_assert( cublasCaxpy ( params->CU.cublasHandle, m12, &alpha, psi, 1, I_d, 1) );
			}

			else if ( params->IM.mode == 2 )
			{
				diffractionPattern( psi, k, params, params_d );
				cublas_assert( cublasCaxpy ( params->CU.cublasHandle, m12, &alpha, psi, 1, I_d, 1) );
			}
		}

 		if( printLevel > 1 )
		{    saveComplex2D(exitwave_d, params, k, exitwave); }

 		addNoiseAndMtf( J_d, I_d, pD, k, poissonState_d, params, params_d );
	}

	cuda_assert ( cudaMemcpy ( image, J_d, n123 * sizeof ( float ), cudaMemcpyDeviceToHost ) );

	// Calculate and save the untilted potential
	k--;
	// Avoid a double calculation for the common case of frPh == 0 and no tilts

	if(printLevel > 0)
	{
		potential  = (float *) malloc (2* m123*sizeof(float));

		if ( ( subSlTh > 1.f ) || ( frPh > 0 ) || ( ( abs( params->IM.tiltspec[2 * k] ) > FLT_EPSILON ) || ( abs( params->IM.tiltspec[2 * k + 1] ) > FLT_EPSILON  ) ) )
		{
			setSubSlices( params, params_d, 1.f / subSlTh );
			cufftHandle cufftPlanBatchBis;
			setCufftPlanBatch( &cufftPlanBatchBis, params );
			for (  s = 0; s < params->IM.m3; s++ )
			{
				phaseGrating( V_d, nAt, nZ, params, params_d, xyzCoordTO_d, imPot, Z_d, Zlist, occ_d, cufftPlanBatchBis,s);
				saveComplex2D(V_d, params, s, potential);
			}
			cufft_assert ( cufftDestroy ( cufftPlanBatchBis ) );
		}
	}

	writeBinary ( image_name, image,  n123 );


	writeHdf5 (emd_name, image, potential, exitwave, params, &Z_d, &xyzCoord_d, &DWF_d, &occ_d);

	cuda_assert ( cudaFree ( xyzCoordTO_d ) );
	cuda_assert ( cudaFree ( xyzCoordFP_d ) );
	cuda_assert ( cudaFree ( V_d ) );
	cuda_assert ( cudaFree ( I_d ) );
	cuda_assert ( cudaFree ( J_d ) );
	cuda_assert ( cudaFree ( dwfState_d ) );
	cuda_assert ( cudaFree ( poissonState_d ) );
	free( Zlist );
	freeDeviceParams ( &params_d, params->IM.n3 );
	cufft_assert ( cufftDestroy ( cufftPlanBatch ) );
	free(image);

	if(printLevel > 0)
	{
		free(potential);
	}
	if(printLevel > 1)
	{
		free(exitwave);
	}
}


void tiltCoordinates( float* xyz_d, int nAt, float t_0, float t_1, float t_2, params_t* params )
{
	float s, c;

	// Counter clockwise rotation about the 3rd axis (z-axis)
	if ( abs( t_2 ) > FLT_EPSILON )
	{
		c =  cosf( t_2 );
		s = -sinf( t_2 ); // Minus because we want a COUNTER CLOCKWISE rotation, while CUBLAS does a CLOCKWISE rotation
		cublas_assert( cublasSrot (params->CU.cublasHandle, nAt, &( xyz_d[0] ), 3, &( xyz_d[1] ), 3, &c, &s ) );
	}

	// Counter clockwise rotation about the 1st axis (y-axis)
	if ( abs( t_1 ) > FLT_EPSILON )
	{
		c =  cosf( t_1 );
		s = -sinf( t_1 ); // Minus because we want a COUNTER CLOCKWISE rotation, while CUBLAS does a CLOCKWISE rotation
		cublas_assert( cublasSrot (params->CU.cublasHandle, nAt, &( xyz_d[0] ), 3, &( xyz_d[2] ), 3, &c, &s ) );
	}

	// Counter clockwise rotation about the 2nd axis (x-axis)
	if ( abs( t_0 ) > FLT_EPSILON )
	{
		c =  cosf( t_0 );
		s = -sinf( t_0 ); // Minus because we want a COUNTER CLOCKWISE rotation, while CUBLAS does a CLOCKWISE rotation
		cublas_assert( cublasSrot (params->CU.cublasHandle, nAt, &( xyz_d[1] ), 3, &( xyz_d[2] ), 3, &c, &s ) );
	}
}

void atomJitter( float* xyz_d, curandState* state, int nAt, float* DWF_d )
{
	// Apply the Debye-Waller Factor (DWF) to the atom positions
	// DWF = 8*pi*pi*<u*u>, with u the deviation from the equilibrium
	// So the variance of the normal distr. is <u*u> = DWF/(8*pi*pi)
	atomJitter_d <<< myGSize( 3 * nAt ), myBSize( 3 * nAt ) >>> ( xyz_d, DWF_d, nAt, state );
}

void myGBSize( int* gbS, int size )
{
    int dev =gpu_index;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    const int maxGS = deviceProp.maxGridSize[0]/2 ; // HALF of max gridsize allowed by device, it is taken double elsewhere
    const int maxBS = deviceProp.maxThreadsDim[0]; // Maximum blocksize allowed by device.

    int bS = maxBS;
    int gS = size / bS + 1;

    if ( gS > maxGS )
    {    gS = maxGS; }

    if ( bS > maxBS )
    {    bS = maxBS; }

	if ( ( bS * gS ) < size )
    {    fprintf ( stderr, "    WARNING: Dimensions of the object too large for the GPU." ); }

	gbS[0] = gS;
	gbS[1] = bS;
}

int myGSize( int size )
{
	int* gbS;
	gbS = ( int* ) malloc ( 2 * sizeof ( int ) );
	myGBSize( gbS, size );
	return( gbS[0] );
}

int myBSize( int size )
{
	int* gbS;
	gbS = ( int* ) malloc ( 2 * sizeof ( int ) );
	myGBSize( gbS, size );
	return( gbS[1] );
}


void phaseGrating( cufftComplex* V_d, int nAt, int nZ, params_t* params, params_t* params_d, float* xyz_d, float imPot, int* Z_d, int* Zlist, float* occ_d, cufftHandle cufftPlanB, int s )
{
	cufftComplex *V1_d, *V2_d, alpha;
	int gS = params->CU.gS, gS2D = params->CU.gS2D, bS = params->CU.bS;

	int m12 = params->IM.m1 * params->IM.m2;

	alpha.x = 1.f;
	alpha.y = 0.f;
	cuda_assert ( cudaMalloc ( ( void** ) &V1_d, m12 * sizeof ( cufftComplex ) ) ); // V1_d m123
	cuda_assert ( cudaMalloc ( ( void** ) &V2_d, m12 * sizeof ( cufftComplex ) ) );

	initialValues <<< gS * 2, bS >>> ( V_d,  m12, 0.f, 0.f );
	for ( int j = 0; j < nZ; j++ )
	{
		initialValues <<< gS * 2, bS >>> ( V1_d, m12, 0.f, 0.f );
 		squareAtoms_d <<< myGSize( nAt ), myBSize( nAt ) >>> ( V1_d, params_d, nAt, Z_d, Zlist[j], xyz_d, imPot, occ_d,s );
		projectedPotential_d <<< gS2D, bS >>> ( V2_d, Zlist[j], params_d );

		divideBySinc <<< gS2D, bS >>> ( V2_d, params_d );
		cufft_assert( cufftExecC2C( cufftPlanB, V1_d, V1_d, CUFFT_FORWARD ) );

		multiplyWithProjectedPotential_d <<< gS, bS >>> ( V1_d, V2_d, params_d );

		cufft_assert( cufftExecC2C( cufftPlanB, V1_d, V1_d, CUFFT_INVERSE ) );
		cublas_assert( cublasCaxpy( params->CU.cublasHandle, m12, &alpha, V1_d, 1, V_d, 1) );
	}
	cuda_assert ( cudaFree ( V1_d ) );
	cuda_assert ( cudaFree ( V2_d ) );
}


int listOfElements( int* Zlist, int nAt, int *Z_d )
{
	int nZ = 1, flag, *Z_h;

	Z_h = ( int* ) calloc( nAt, sizeof( int ) );
	cuda_assert ( cudaMemcpy ( Z_h, Z_d, nAt * sizeof ( int ), cudaMemcpyDeviceToHost ) );

	for ( int j = 0; j < 103; j++ )
	{    Zlist[j] = 0; }


	Zlist[0] = Z_h[0];

	int j, k;
	for (  j = 1; j < nAt; j++ )
	{
		flag = 0;
		for (  k = 0; k < nZ; k++ )
		{
			if ( Zlist[k] == Z_h[j] )
			{    flag = 1; }
		}
		if ( flag == 0 )
		{
			Zlist[nZ] = Z_h[j];
			nZ++;
		}
	}

	free( Z_h );
	return( nZ );
}


void setCufftPlanBatch( cufftHandle* plan, params_t* params )
{
	cufft_assert ( cufftPlan2d (plan, params->IM.m1, params->IM.m2, CUFFT_C2C ) );
}


void addNoiseAndMtf( float* J_d, cufftComplex* I_d, float dose, int k, curandState* poissonState, params_t* params, params_t* params_d )
{
	const int m1 = params->IM.m1;
	const int m2 = params->IM.m2;
	const int gS2D = params->CU.gS2D;
	const int bS = params->CU.bS;
	float alpha;

	alpha = 1.f / ( (float) ( m1 * m2 ) );

	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, I_d, I_d, CUFFT_FORWARD ) );

	if ( fabsf( params->EM.illangle ) > FLT_EPSILON  )
	{
		if( params->IM.mode == 0 )
		{	multiplySpatialIncoherence <<< gS2D, bS >>> ( I_d, k, params_d ); }
		if( ( params->IM.mode == 1 ) || ( params->IM.mode == 2 ) )
		{	multiplySpatialIncoherenceDP <<< gS2D, bS >>> ( I_d, k, params_d ); }
	}

	if ( dose > FLT_EPSILON )
	{
		cublas_assert ( cublasCsscal (params->CU.cublasHandle, m1 * m2, &alpha, I_d, 1 ) );
		cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, I_d, I_d, CUFFT_INVERSE ) );
		ascombeNoise_d <<< gS2D, bS >>> ( I_d, dose, m1 * m2, poissonState );

		cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, I_d, I_d, CUFFT_FORWARD ) );
	}

	multiplyMtf <<< gS2D, bS >>> ( I_d, params_d );
	cublas_assert ( cublasCsscal (params->CU.cublasHandle, m1 * m2, &alpha, I_d, 1 ) );
	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, I_d, I_d, CUFFT_INVERSE ) );

	copyMiddleOut <<< gS2D, bS >>> ( &( J_d[k * params->IM.n1 * params->IM.n2] ),  I_d, params_d );
}


void saveMeasurements( float* J_d, params_t* params )
{
	int n123 = params->IM.n1 * params->IM.n2 * params->IM.n3;
	float *J_h;

	J_h = ( float* ) malloc ( n123 * sizeof ( float ) );
	cuda_assert ( cudaMemcpy ( J_h, J_d, n123 * sizeof ( float ), cudaMemcpyDeviceToHost ) );

	writeBinary ( "Measurements.bin", J_h,  n123 );

	free ( J_h );
}


void savePotential( cufftComplex* V_d, params_t* params )
{
	const int m123  = params->IM.m1 * params->IM.m2 * params->IM.m3;
	cufftComplex* V_h;
	float* Vri_h;

	V_h = ( cufftComplex* ) malloc ( m123 * sizeof ( cufftComplex ) );
	Vri_h = ( float* ) malloc ( m123 * sizeof ( float ) );

	cuda_assert ( cudaMemcpy ( V_h, V_d, m123 * sizeof ( cufftComplex ), cudaMemcpyDeviceToHost ) );

	realPart ( Vri_h, V_h, m123 );
	writeBinary ( "PotentialRealOriginal.bin", Vri_h, m123 );

	imagPart ( Vri_h, V_h, m123 );
	writeBinary ( "PotentialImagOriginal.bin", Vri_h, m123 );

	free( V_h );
	free( Vri_h );
}

void savePotential2D( cufftComplex* V_d, params_t* params, int s )
{
	char sSave[128];
	const int m12  = params->IM.m1 * params->IM.m2;
	cufftComplex* V_h;
	float* Vri_h;

	V_h = ( cufftComplex* ) malloc ( m12 * sizeof ( cufftComplex ) );
	Vri_h = ( float* ) malloc ( m12 * sizeof ( float ) );

	cuda_assert ( cudaMemcpy ( V_h, V_d, m12 * sizeof ( cufftComplex ), cudaMemcpyDeviceToHost ) );

	sprintf(sSave,"test-real/CBED-real-%02d.bin",s);
	realPart ( Vri_h, V_h, m12 );
	writeBinary (sSave, Vri_h, m12 );

	sprintf(sSave,"test-img/CBED-imag-%02d.bin",s);
	imagPart ( Vri_h, V_h, m12 );
	writeBinary ( sSave, Vri_h, m12 );

	free( V_h );
	free( Vri_h );
}

void saveComplex2D( cufftComplex* V_d, params_t* params, int s, float * f )
{
	const int m12  = params->IM.m1 * params->IM.m2;
	cufftComplex* V_h;
	float* Vri_h;

	V_h = ( cufftComplex* ) malloc ( m12 * sizeof ( cufftComplex ) );
	Vri_h = ( float* ) malloc ( m12 * sizeof ( float ) );

	cuda_assert ( cudaMemcpy ( V_h, V_d, m12 * sizeof ( cufftComplex ), cudaMemcpyDeviceToHost ) );

	realPart ( Vri_h, V_h, m12 );

	for(int i= 0; i< m12; i++)
	{	  f[s*2*m12 + 2*i] = Vri_h[i]; }


	imagPart ( Vri_h, V_h, m12 );
	for(int i= 0; i< m12; i++)
	{    f[s*2*m12 + 2*i+1] = Vri_h[i];	}

	free( V_h );
	free( Vri_h );
}

void diffractionPattern( cufftComplex* psi, int k, params_t* params, params_t* params_d )
{
	const int m12 = params->IM.m1 * params->IM.m2;
	float alpha = sqrtf( 1.f / ( (float) m12 ) );

	if ( params->IM.doBeamTilt )
	{    tiltBeam_d <<< params->CU.gS, params->CU.bS >>> ( psi, k, params_d, -1 ); }

	if( params->IM.mode == 1 )
	{
		applyMaskFiltering( psi, params, params_d );
		bandwidthLimit ( psi, params );
	}

	cufft_assert( cufftExecC2C( params->CU.cufftPlan, psi, psi, CUFFT_FORWARD ) );
	cufftShift2D_h( psi, params->IM.m1, params->IM.m2, params->CU.bS );
	cublas_assert ( cublasCsscal( params->CU.cublasHandle, m12, &alpha, psi, 1 ) );
	intensityValues <<< params->CU.gS2D, params->CU.bS>>> ( psi, m12 );
}

float subSliceRatio( float slice, float subSlice )
{
	float ratio = 1.f;

	if ( ( subSlice > 1e-12f ) && ( subSlice < slice ) ) // subSlice must be larger than 0.01 Angstrom
	{    ratio = ceilf( slice / subSlice ); }

	return( ratio );
}

void setSubSlices( params_t* params, params_t* params_d, float ratio )
{
	params->IM.m3 = (int) ( ( (float) params->IM.m3 ) * ratio );
	params->IM.d3 /= ratio;
	params->CU.bS = myBSize( params->IM.m1 * params->IM.m2 * params->IM.m3 );
	params->CU.gS = myGSize( params->IM.m1 * params->IM.m2);
	params->CU.gS2D = myGSize( params->IM.m1 * params->IM.m2 );

	cuda_assert ( cudaMemcpy ( &( params_d->IM.m3 ), &( params->IM.m3 ), sizeof ( int ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( &( params_d->IM.d3 ), &( params->IM.d3 ), sizeof ( float ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( &( params_d->CU.bS ), &( params->CU.bS ), sizeof ( int ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( &( params_d->CU.gS ), &( params->CU.gS ), sizeof ( int ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( &( params_d->CU.gS2D ), &( params->CU.gS2D ), sizeof ( int ), cudaMemcpyHostToDevice ) );
}
