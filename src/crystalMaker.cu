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
		xyz_d[i] += x * 0.11253954f * sqrtf( dwf_d[i/3] ); // 0.11 = 1/(pi*sqrt(8))
		state[i] = localState;  // Copy (modified) state back to global memory
	}
}

__global__ void ascombeNoise_d( cufftComplex* f, float dose, int size, curandState *state )
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


__global__ void squareAtoms_d ( cufftComplex* V, params_t* params, int nAt, int* Z, int Z0, float* xyz, float imPot, float* occ, int zlayer )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i < nAt ) {
		if ( Z[i] == Z0 ) {
			const int m1 = params->IM.m1;
			const int m2 = params->IM.m2;
			const int m3 = params->IM.m3;
			float x1, x2, w = 1.f;
			x1 = xyz[i * 3 + 0] / params->IM.d1 + ( (float) m1 ) * 0.5f - 0.5f;
			x2 = xyz[i * 3 + 1] / params->IM.d2 + ( (float) m2 ) * 0.5f - 0.5f;
			int i3 = (int) ( roundf( xyz[i * 3 + 2] / params->IM.d3 + ( (float) m3 ) * 0.5f - 0.5f ) );

			if ( ( ( x1 > 1.f ) && ( x1 < ( (float) ( m1 - 2 ) ) ) )  &&  ( ( x2 > 1.f ) 
				&& ( x2 < ( (float) ( m2 - 2 ) ) ) )  &&  ( i3 == zlayer ) ) {

				if ( x1 < ( (float) params->IM.dn1 ) ) { // Compute the weight w as a cosine. Occupancy goes to zero in the boarders
					w *= 0.5f * cosf( ( x1 - ( (float) params->IM.dn1 ) ) / ( (float) params->IM.dn1 ) * params->cst.pi ) + 0.5f;
				} else { if ( x1 > ( (float) params->IM.dn1 + params->IM.n1 ) ) {
					w *= 0.5f * cosf( ( x1 - ( (float) params->IM.dn1 + params->IM.n1 ) ) / ( (float) params->IM.dn1 ) * params->cst.pi ) + 0.5f ;
				} }
				if ( x2 < ( (float) params->IM.dn2 ) ) {
					w *= 0.5f * cosf( ( x2 - ( (float) params->IM.dn2 ) ) / ( (float) params->IM.dn2 ) * params->cst.pi ) + 0.5f ;
				} else { if ( x2 > ( (float) params->IM.dn2 + params->IM.n2 ) ) {
					w *= 0.5f * cosf( ( x2 - ( (float) params->IM.dn2 + params->IM.n2 ) ) / ( (float) params->IM.dn2 ) * params->cst.pi ) + 0.5f ;
				} }

				int i1 = (int) roundf( x1 );
				int i2 = (int) roundf( x2 );
				int j;
				float r1 = x1 - ( (float) i1 );
				float r2 = x2 - ( (float) i2 );
				float temp;

				sgCoord(j, i1, i2, m1);
				temp = ( ( 1 - fabsf( r1 ) ) * ( 1 - fabsf( r2 ) ) ) * ( occ[i] * w );
				atomicAdd( &( V[j].x ), temp );
				atomicAdd( &( V[j].y ), temp * imPot );

				i2 += mySignum_d( r2 );
				sgCoord(j, i1, i2, m1 );
				temp = ( ( 1 - fabsf( r1 ) ) * fabsf( r2 ) ) * ( occ[i] * w );
				atomicAdd( &( V[j].x ), temp );
				atomicAdd( &( V[j].y ), temp * imPot );

				i1 += mySignum_d( r1 );
				sgCoord(j, i1, i2, m1 );
				temp = ( fabsf( r1 ) * fabsf( r2 ) ) * ( occ[i] * w );
				atomicAdd( &( V[j].x ), temp );
				atomicAdd( &( V[j].y ), temp * imPot );

				i2 -= mySignum_d( r2 );
				sgCoord(j, i1, i2, m1 );
				temp = ( fabsf( r1 ) * ( 1 - fabsf( r2 ) ) ) * ( occ[i] * w );
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
	
	if ( i < m1 * m2 ) {
		float V2x = V2[i].x;
		V1[i].x *= V2x;
		V1[i].y *= V2x;
	}
}

__global__ void shiftCoordinates_d ( float* xyzCoord_d, float sX, float sY, int size )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ( i < size )	{
		xyzCoord_d[i*3]   += sX;
		xyzCoord_d[i*3+1] += sY;
	}
}

__global__ void multiplyDetMask_d( float* J_d, params_t* params )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	int i1, i2;
	float sinth1, sinth2;
	const int m1 = params->IM.m1;
	const int m2 = params->IM.m2;

	if ( i < m1 * m2 ) {
		dbCoord ( i1, i2, i, m1 );
		owCoordIp ( i1, m1 );
		owCoordIp ( i2, m2 );
		sinth1 = ( ( ( float ) i1 ) / ( ( float ) m1 ) ) * ( params->EM.lambda / params->IM.d1 );
		sinth2 = ( ( ( float ) i2 ) / ( ( float ) m2 ) ) * ( params->EM.lambda / params->IM.d2 );
		sinth2 = sqrtf( sinth1 * sinth1 + sinth2 * sinth2 );
		if ( ( sinth2 < sinf( params->SCN.thIn ) ) || ( sinth2 > sinf( params->SCN.thOut ) ) ){
			J_d[i] = 0.f;
		}
	}
}


void buildMeasurements( params_t* params, int* Z_d, float* xyzCoord_d, float* DWF_d, float* occ_d )
{
	float *xyzCoordTO_d, *J_h;
	cufftComplex *I_d, *Isum_d, alpha;
	int count, m12n3, m12, o12, nZ, *Zlist, k, itCnt = 0;;
	curandState *dwfState_d, *poissonState_d;
	params_t *params_d;
	int nAt = params->SAMPLE.nAt;
	int frPh = params->IM.frPh;
	float pD = params->IM.pD;
	float imPot = params->SAMPLE.imPot;
	float subSlTh = params->IM.subSlTh; 

	setDeviceParams ( &params_d, params );
	subSlTh = subSliceRatio( params->IM.d3, subSlTh );
	setSubSlices( params, params_d, subSlTh );

	count = 1;
	m12n3 = params->IM.m1 * params->IM.m2 * params->IM.n3;
	if ( params->IM.mode == 3 ) {
		m12n3 = params->SCN.o1 * params->SCN.o2 * params->IM.n3;
	}
	m12 = params->IM.m1 * params->IM.m2;
	o12 = params->SCN.o1 * params->SCN.o2;
	k = 0;

	cuda_assert ( cudaMalloc ( ( void** ) &xyzCoordTO_d, nAt * 3 * sizeof ( float ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &I_d, m12 * sizeof ( cufftComplex ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &Isum_d, m12 * sizeof ( cufftComplex ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &dwfState_d, 3 * nAt * sizeof( curandState ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &poissonState_d, m12 * sizeof( curandState ) ) );
	Zlist = ( int* ) malloc ( 103 * sizeof ( int ) );
	J_h = ( float* ) malloc ( m12n3 * sizeof ( float ) );
	
	cublas_assert ( cublasScopy ( params->CU.cublasHandle, nAt * 3, xyzCoord_d, 1, xyzCoordTO_d, 1 ) );
	tiltCoordinates( xyzCoordTO_d, nAt, params->IM.specimen_tilt_offset_x, params->IM.specimen_tilt_offset_y,
		params->IM.specimen_tilt_offset_z, params ); // Create coordinates with the tilt offset.

	nZ = listOfElements( Zlist, nAt, Z_d );

	if ( frPh > 0 )	{
		setupCurandState_d <<< myGSize( 3 * nAt ), myBSize( 3 * nAt ) >>> ( dwfState_d, 1, 3 * nAt );
		cuda_assert ( cudaDeviceSynchronize () );
	}
	if ( pD > FLT_MIN ) {
		setupCurandState_d <<< myGSize( m12 ), myBSize( m12 ) >>> ( poissonState_d, 1 + params->IM.n3, m12 ); 
		cuda_assert ( cudaDeviceSynchronize () );
	}
	if ( frPh > 0 )	{   
		count = frPh; 
	}
	alpha.x = 1.f / ( (float) count );
	alpha.y = 0.f;
	
	// Calculate and save the measurements
	for ( k = 0; k < params->IM.n3; k++ ) {
		cublas_assert ( cublasScopy ( params->CU.cublasHandle, nAt * 3, xyzCoordTO_d, 1, xyzCoord_d, 1 ) );
		tiltCoordinates( xyzCoord_d, nAt, params->IM.tiltspec[2 * k], params->IM.tiltspec[2 * k + 1], 0.f, params );
		for ( int scnI = 0; scnI < o12; scnI++ ) {
			initialValues <<< params->CU.gS2D * 2, params->CU.bS >>> ( Isum_d, m12, 0.f, 0.f );
			cuda_assert ( cudaDeviceSynchronize () );
			shiftCoordinates( xyzCoord_d, params, scnI );
			for ( int j = 0; j < count; j++ ) {
				initialValues <<< params->CU.gS2D * 2, params->CU.bS >>> ( I_d, m12, 0.f, 0.f );
				cuda_assert ( cudaDeviceSynchronize () );
				rawMeasurement( I_d, params, params_d, k, nAt, nZ, xyzCoord_d, imPot, Z_d, Zlist, DWF_d, occ_d, dwfState_d, &itCnt );
				cublas_assert( cublasCaxpy( params->CU.cublasHandle, m12, &alpha, I_d, 1, Isum_d, 1) );
			}
			addNoiseAndMtf( Isum_d, pD, k, poissonState_d, params, params_d );
			integrateRecordings( J_h, Isum_d, params, params_d, k, scnI );
		}
	}
	writeBinary ( "Measurements.bin", J_h,  m12n3 );

	cuda_assert ( cudaFree ( xyzCoordTO_d ) );
	cuda_assert ( cudaFree ( I_d ) );
	cuda_assert ( cudaFree ( Isum_d ) );
	cuda_assert ( cudaFree ( dwfState_d ) );
	cuda_assert ( cudaFree ( poissonState_d ) );

	free( Zlist );
	freeDeviceParams ( &params_d, params->IM.n3 );
	free( J_h );
}

void tiltCoordinates( float* xyz_d, int nAt, float t_0, float t_1, float t_2, params_t* params )
{
	float s, c;

	// Counter clockwise rotation about the 3rd axis (z-axis)
	if ( abs( t_2 ) > FLT_EPSILON )	{
		c =  cosf( t_2 );
		s = -sinf( t_2 ); // Minus because we want a COUNTER CLOCKWISE rotation, while CUBLAS does a CLOCKWISE rotation
		cublas_assert( cublasSrot (params->CU.cublasHandle, nAt, &( xyz_d[0] ), 3, &( xyz_d[1] ), 3, &c, &s ) );
	}
	// Counter clockwise rotation about the 2nd dimension (x-axis) This is tilt_y, sorry for the misnomer
	if ( abs( t_1 ) > FLT_EPSILON )	{
		c =  cosf( t_1 );
		s = -sinf( t_1 ); // Minus because we want a COUNTER CLOCKWISE rotation, while CUBLAS does a CLOCKWISE rotation
		cublas_assert( cublasSrot (params->CU.cublasHandle, nAt, &( xyz_d[0] ), 3, &( xyz_d[2] ), 3, &c, &s ) );
	}
	// Counter clockwise rotation about the 1st dimension (y-axis) This is tilt_x, sorry for the misnomer
	if ( abs( t_0 ) > FLT_EPSILON )	{
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
	const int maxGS = 1073741823; // HALF of max gridsize allowed by device, it is taken double elsewhere
    const int maxBS = 1024; // Maximum blocksize allowed by device.

    int bS = 1024;
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


void phaseGrating( cufftComplex* V_d, int nAt, int nZ, params_t* params, params_t* params_d, float* xyz_d, float imPot, int* Z_d, int* Zlist, float* occ_d, int zlayer )
{
	cufftComplex *V1_d, *V2_d, alpha;
	int gS2D = params->CU.gS2D;
	int bS = params->CU.bS;
	int m12 = params->IM.m1 * params->IM.m2;
	
	alpha.x = 1.f; 
	alpha.y = 0.f;
	cuda_assert ( cudaMalloc ( ( void** ) &V1_d, m12 * sizeof ( cufftComplex ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) &V2_d, m12 * sizeof ( cufftComplex ) ) );

	initialValues <<< gS2D * 2, bS >>> ( V_d,  m12, 0.f, 0.f );

	for ( int j = 0; j < nZ; j++ )
	{
		initialValues <<< gS2D * 2, bS >>> ( V1_d, m12, 0.f, 0.f );
		squareAtoms_d <<< myGSize( nAt ), myBSize( nAt ) >>> ( V1_d, params_d, nAt, Z_d, Zlist[j], xyz_d, imPot, occ_d, zlayer );
		projectedPotential_d <<< gS2D, bS >>> ( V2_d, Zlist[j], params_d );
		divideBySinc <<< gS2D, bS >>> ( V2_d, params_d );
		cufft_assert( cufftExecC2C( params->CU.cufftPlan, V1_d, V1_d, CUFFT_FORWARD ) );
		multiplyElementwise <<< gS2D, bS >>> ( V1_d, V2_d, m12 );
		cufft_assert( cufftExecC2C( params->CU.cufftPlan, V1_d, V1_d, CUFFT_INVERSE ) );
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

	for (int j = 0; j < 103; j++ )
	{    Zlist[j] = 0; }
	Zlist[0] = Z_h[0];

	for ( int j = 1; j < nAt; j++ )
	{
		flag = 0;
		for ( int k = 0; k < nZ; k++ )
		{
			if ( Zlist[k] == Z_h[j] )
			{    flag = 1; }
		}
		if (flag == 0)
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
	int *batchDims, m12;

	m12 = params->IM.m1 * params->IM.m2;
	batchDims = ( int* ) malloc ( 2 * sizeof ( int ) );
	
	batchDims[0] = params->IM.m1;
	batchDims[1] = params->IM.m2;
	
	cufft_assert( cufftPlanMany ( plan, 2, batchDims, batchDims, 1, m12, batchDims, 1, m12, CUFFT_C2C, params->IM.m3 ) );

	free ( batchDims );
}

void addNoiseAndMtf( cufftComplex* I_d, float dose, int k, curandState* poissonState, params_t* params, params_t* params_d )
{
	const int m1 = params->IM.m1;
	const int m2 = params->IM.m2;
	const int gS2D = params->CU.gS2D;
	const int bS = params->CU.bS;
	int mode = params->IM.mode;
	float alpha;

	alpha = 1.f / ( (float) ( m1 * m2 ) );

	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, I_d, I_d, CUFFT_FORWARD ) );
	if ( mode == 0 ) {
		multiplySpatialIncoherence <<< gS2D, bS >>> ( I_d, k, params_d, 1 );
	}
	if ( ( dose > FLT_EPSILON ) && ( ( mode == 0 ) || ( mode == 1 ) || ( mode == 2 ) ) ) {
		cublas_assert ( cublasCsscal (params->CU.cublasHandle, m1 * m2, &alpha, I_d, 1 ) );
		cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, I_d, I_d, CUFFT_INVERSE ) );
		ascombeNoise_d <<< gS2D, bS >>> ( I_d, dose, m1 * m2, poissonState ); 
		cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, I_d, I_d, CUFFT_FORWARD ) );
	}
	if ( ( mode == 0 ) || ( mode == 2 ) ){
		multiplyMtf <<< gS2D, bS >>> ( I_d, params_d, 1 );
	}
	/*if ( mode == 1 ){ // No MTF for mode == 1, the periodic boundary conditions mess up the DP
		multiplyMtf <<< gS2D, bS >>> ( I_d, params_d, 1 );
	}*/
	cublas_assert ( cublasCsscal (params->CU.cublasHandle, m1 * m2, &alpha, I_d, 1 ) );
	cufft_assert ( cufftExecC2C ( params->CU.cufftPlan, I_d, I_d, CUFFT_INVERSE ) );
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
	params->CU.gS = myGSize( params->IM.m1 * params->IM.m2 * params->IM.m3 );
	params->CU.gS2D = myGSize( params->IM.m1 * params->IM.m2 );

	cuda_assert ( cudaMemcpy ( &( params_d->IM.m3 ), &( params->IM.m3 ), sizeof ( int ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( &( params_d->IM.d3 ), &( params->IM.d3 ), sizeof ( float ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( &( params_d->CU.bS ), &( params->CU.bS ), sizeof ( int ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( &( params_d->CU.gS ), &( params->CU.gS ), sizeof ( int ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( &( params_d->CU.gS2D ), &( params->CU.gS2D ), sizeof ( int ), cudaMemcpyHostToDevice ) );
}

void farFieldTransform( cufftComplex* psi, params_t* params, params_t* params_d )
{ 
	const int m12 = params->IM.m1 * params->IM.m2;
	float alpha = sqrtf( 1.f / ( (float) m12 ) );
	
	cufft_assert( cufftExecC2C( params->CU.cufftPlan, psi, psi, CUFFT_FORWARD ) );
	zeroHighFreq <<< 2 * params->CU.gS2D, params->CU.bS >>> ( psi, params->IM.m1, params->IM.m2 );
	cufftShift2D_h( psi, params->IM.m1, params->IM.m2, params->CU.bS );
	cublas_assert ( cublasCsscal( params->CU.cublasHandle, m12, &alpha, psi, 1 ) );
}

void shiftCoordinates( float* xyzCoord_d, params_t* params, int scnI )
{
	// Applies an absolute translation to xyzCoords in the first iteration (scnI == 0)
	// and increments for further iterations (scnI > 0)
	float s0, s1;
	int i0 = 0, i1 = 0, j0, j1, nAt;
	nAt = params->SAMPLE.nAt;

	dbCoord( j0, j1, scnI, params->SCN.o1 );
	owCoordIp( j0, params->SCN.o1 );
	owCoordIp( j1, params->SCN.o2 );

	if ( scnI > 0 ) { // Calculate the previous position and apply a relative step
		dbCoord( i0, i1, scnI-1, params->SCN.o1 );
		owCoordIp( i0, params->SCN.o1 );
		owCoordIp( i1, params->SCN.o2 );
	}
	s0 = ( (float) ( j0 - i0 ) ) * params->SCN.dSx;
	s1 = ( (float) ( j1 - i1 ) ) * params->SCN.dSy;
	if ( scnI == 0 ) { // Apply an absolute step in the beginning
		s0 += params->SCN.c1;
		s1 += params->SCN.c2;
	}
	s0 *= -1.f; // Atoms have to be shifted opposite of the beam
	s1 *= -1.f;

	shiftCoordinates_d <<<  myGSize( nAt ), myBSize( nAt ) >>> ( xyzCoord_d, s0, s1, nAt );
	cuda_assert ( cudaDeviceSynchronize () );
}

void integrateRecordings( float* J_h, cufftComplex* J_d, params_t* params, params_t* params_d, int k, int scnI )
{
	int m12, o12, bS, gS;
	m12 = params->IM.m1 * params->IM.m2;
	o12 = params->SCN.o1 * params->SCN.o2;
	bS = params->CU.bS;
	gS = params->CU.gS2D;
	float tmp = 0.f, *Jtmp_d;

	cuda_assert ( cudaMalloc ( ( void** ) &Jtmp_d, m12 * sizeof ( float ) ) );
	
	cublas_assert( cublasScopy( params->CU.cublasHandle, m12, (float*) J_d, 2, Jtmp_d, 1 ) ); // Copy the real parts of J_d into Jtmp_d

	if ( params->IM.mode != 3 ){
		cuda_assert ( cudaMemcpy ( &(J_h[k*m12]), Jtmp_d,  m12 * sizeof ( float ), cudaMemcpyDeviceToHost ) );
	} else { // i.e. mode == 3
		multiplyDetMask_d <<<  gS, bS >>> ( Jtmp_d, params_d );
		cuda_assert ( cudaDeviceSynchronize () );
		cublas_assert ( cublasSasum( params->CU.cublasHandle, m12, Jtmp_d, 1, &tmp ) );
		J_h[scnI + k*o12] = tmp;
	}
	cuda_assert( cudaFree( Jtmp_d ) );
}
