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

==================================================================*/

#include "complexMath.h"


__global__ void realPartDevice ( float* dst, cuComplex* src, int size )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    { dst[i] = src[i].x; }
}

__global__ void imagPartDevice ( float* dst, cuComplex* src, int size )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    { dst[i] = src[i].y; }
}

__global__ void multiplyElementwise ( cufftComplex* f0, cufftComplex* f1, int size )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    {
        float a, b, c, d;
        a = f0[i].x;
        b = f0[i].y;
        c = f1[i].x;
        d = f1[i].y;
        float k;
        k = a * ( c + d );
        d *= a + b;
        c *= b - a;
        f0[i].x = k - d;
        f0[i].y = k + c;
    }
}

__global__ void initialValues ( cuComplex* V, int size, float initRe, float initIm )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < 2 * size )
    {
        if ( ( i % 2 ) == 0 )
        { V[i / 2].x = initRe; }

        else
        { V[i / 2].y = initIm; }
    }
}

__global__ void multiplyElementwiseFast ( cufftComplex* f0, cufftComplex* f1, int size )
{
    __shared__ float a[64];
    __shared__ float b[64];
    __shared__ float c[64];
    __shared__ float d[64];

    int M = 64;

    if ( ( ( blockIdx.x + 1 ) * blockDim.x ) > ( 4 * size ) )
    { M = ( ( 4 * size ) - ( blockIdx.x * blockDim.x ) ) / 4; }

    int i, k;

    if ( threadIdx.x < 2 * M )
    {
        k = threadIdx.x;
        i = ( blockIdx.x * blockDim.x ) / 4 + k / 2;

        if ( k % 2 == 0 )
        { a[k / 2] = f0[i].x; }

        else
        { b[k / 2] = f0[i].y; }
    }

    else if ( threadIdx.x < 4 * M )
    {
        k = threadIdx.x - 2 * M;
        i = ( blockIdx.x * blockDim.x ) / 4 + k / 2;

        if ( k % 2 == 0 )
        { c[k / 2] = f1[i].x; }

        else
        { d[k / 2] = f1[i].y; }
    }

    __syncthreads ();

    if ( threadIdx.x < 2 * M )
    {
        k = threadIdx.x / 2;
        i = ( blockIdx.x * blockDim.x ) / 4 + k;

        if ( threadIdx.x % 2 == 0 )
        { f0[i].x = a[k] * c[k] - b[k] * d[k]; } // It's the d-variable that keeps on giving troubles, look into it.

        else
        { f0[i].y = b[k] * c[k] + a[k] * d[k]; }
    }
}


__global__ void sumElements_d( cuComplex* dst, cuComplex* src, cuComplex thold, int sizeSrc, const int flag, const float mu0 )
{
	// flag == 0 for regular sum, 
	// flag == 1 for the sum of the absolute values
	// flag == 2 for the sum of thresholded values
	// flag == 3 for the sum of thresholded absolute values

	extern __shared__ cuComplex sum[];
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if ( i < sizeSrc )
	{    
		if ( flag == 3 ) // A sum of thresholded absolute values
		{
			if ( absoluteValue( src[i].x, mu0 ) < thold.x )
			{    sum[threadIdx.x].x = 1.f; }
			else
			{    sum[threadIdx.x].x = 0.f; }

			if ( absoluteValue( src[i].y, mu0 ) < thold.y )
			{    sum[threadIdx.x].y = 1.f; }
			else
			{    sum[threadIdx.x].y = 0.f; }
		}
		else { if ( flag == 2 ) // A sum of thresholded values
		{    
			if ( src[i].x < thold.x )
			{    sum[threadIdx.x].x = 1.f; }
			else
			{    sum[threadIdx.x].x = 0.f; }

			if ( src[i].y < thold.y )
			{    sum[threadIdx.x].y = 1.f; }
			else
			{    sum[threadIdx.x].y = 0.f; }
		}
		else { if ( flag == 1) // it's a sum of absolute values
		{
			sum[threadIdx.x].x = absoluteValue( src[i].x, mu0 );
			sum[threadIdx.x].y = absoluteValue( src[i].y, mu0 );
		}
		else // if flag == 0 it's a normal sum
		{    
			sum[threadIdx.x].x = src[i].x;
			sum[threadIdx.x].y = src[i].y;
		} } }
	}
	else
	{    
		sum[threadIdx.x].x = 0.f;
		sum[threadIdx.x].y = 0.f;
	}
	__syncthreads ();

	int n = blockDim.x/2;
	while ( n > 1 )
	{
		if ( threadIdx.x < n )
		{	
			sum[threadIdx.x].x += sum[threadIdx.x + n].x;
			sum[threadIdx.x].y += sum[threadIdx.x + n].y;
		}
		__syncthreads ();
		n /= 2;
	}

	if ( threadIdx.x == 0 )
	{	
		dst[blockIdx.x].x = sum[0].x + sum[1].x;
		dst[blockIdx.x].y = sum[0].y + sum[1].y;
	}
}


__global__ void addDerivedAbsVals ( cuComplex* dst, cuComplex* src, float fctr, int size, float mu0 )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < 2 * size )
    {
        const int i0 = i / 2;

        if ( ( i % 2 ) == 0 )
        {
            float srcx = src[i0].x;
			if ( ! ( srcx < 0.f ) )
			{    mu0 = 1.f; }

			dst[i0].x += fctr * mu0 * mu0 * srcx / absoluteValue( srcx, mu0 );
        }
        else
        {
            float srcy = src[i0].y;
			if ( ! ( srcy < 0.f ) )
			{    mu0 = 1.f; }
			
			dst[i0].y += fctr * mu0 * mu0 * srcy / absoluteValue( srcy, mu0 );
        }
    }
}


__device__ float absoluteValue( float x, const float mu0 )
{
	float y, epsSq;
	epsSq = 1e-8f;

	if ( x < 0.f )
	{    x *= mu0; }

	y = sqrtf( x * x + epsSq );

	return ( y ); 
}


__global__ void myCaxpy ( cuComplex* y, cuComplex* x, cuComplex a, int size )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    {
		float xRe = x[i].x;
		float xIm = x[i].y;
		float aRe = a.x;
		float aIm = a.y;

		y[i].x += xRe * aRe - xIm * aIm;
		y[i].y += xRe * aIm + xIm * aRe;
    }
}


__global__ void complex_magnitutude_square( cufftComplex *des, cufftComplex *src, int nSize )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < nSize )
	{
		des[i].x =src[i].x*src[i].x+src[i].y*src[i].y ;
		des[i].y = 0.f;
	}
}


__global__ void memcpy_complex_d( cufftComplex *des, cufftComplex *src, int nSize )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < nSize )
	{
		des[i] =src[i];
	}
}

__global__ void copyCufftShift_dim1_d ( cufftComplex* f0, cufftComplex* f1, int dim1, int dim2, int n0, int offSet )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < ( dim1 * n0 ) )
    {
        int i1, i2, j;
        dbCoord( i1, i2, i, dim1 ); 
		sgCoord( j, i1, i2 + offSet , dim1);

		f0[j].x = f1[i].x;
		f0[j].y = f1[i].y;
	}
}

__global__ void copyCufftShift_dim2_d ( cufftComplex* f0, cufftComplex* f1, int dim1, int dim2, int n0, int offSet )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < ( dim2 * n0 ) )
    {
        int i1, i2, j;
        dbCoord( i1, i2, i, n0 ); 
		sgCoord( j, i1 + offSet, i2 , dim1);

		f0[j].x = f1[i].x;
		f0[j].y = f1[i].y;
	}
}


__global__ void cufftShift2D_dim1_d ( cufftComplex* f1, cufftComplex* f0, int dim1, int dim2, int n0, int offSet )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < ( dim1 * n0 ) )
    {
        int i0, i1, i2, j;
        dbCoord ( i1, i2, i, dim1 ); 
  
		if ( i1 < ( dim1 - ( dim1 / 2 ) ) )
		{    sgCoord(j, i1 + dim1/2, i2, dim1); }
		else
		{    sgCoord(j, i1 - ( dim1 - (dim1/2) ), i2, dim1); }

		sgCoord( i0, i1, i2 + offSet, dim1 );

		f1[j].x = f0[i0].x;
		f1[j].y = f0[i0].y;
	}
}

__global__ void cufftShift2D_dim2_d ( cufftComplex* f1, cufftComplex* f0, int dim1, int dim2, int n0, int offSet )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < ( n0 * dim2 ) )
    {
        int i0, i1, i2, j;
        dbCoord ( i1, i2, i, n0 ); 
  
		if ( i2 < ( dim2 - ( dim2 / 2 ) ) )
		{    sgCoord(j, i1, i2 + dim2/2, n0); }
		else
		{    sgCoord(j, i1, i2 - ( dim2 - (dim2/2) ), n0); }

		sgCoord( i0, i1 + offSet, i2, dim1 );

		f1[j].x = f0[i0].x;
		f1[j].y = f0[i0].y;
	}
}

__global__ void cufftIShift2D_dim1_d ( cufftComplex* f1, cufftComplex* f0, int dim1, int dim2, int n0, int offSet )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < ( dim1 * n0 ) )
    {
        int i0, i1, i2, j;
        dbCoord ( i1, i2, i, dim1 ); 
  
		if ( i1 < ( dim1 / 2 ) )
		{    sgCoord(j, i1 + ( dim1 - ( dim1 / 2 ) ), i2, dim1 ); }
		else
		{    sgCoord(j, i1 - ( dim1 / 2 ), i2, dim1); }

		sgCoord( i0, i1, i2 + offSet, dim1 );

		f1[j].x = f0[i0].x;
		f1[j].y = f0[i0].y;
	}
}

__global__ void cufftIShift2D_dim2_d ( cufftComplex* f1, cufftComplex* f0, int dim1, int dim2, int n0, int offSet )
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < ( n0 * dim2 ) )
    {
        int i0, i1, i2, j;
        dbCoord ( i1, i2, i, n0 ); 
  
		if ( i2 < ( dim2 / 2 ) )
		{    sgCoord(j, i1, i2 + ( dim2 - ( dim2 / 2 ) ), n0 ); }
		else
		{    sgCoord(j, i1, i2 - ( dim2 / 2 ), n0); }

		sgCoord( i0, i1 + offSet, i2, dim1 );

		f1[j].x = f0[i0].x;
		f1[j].y = f0[i0].y;
	}
}

__global__ void addConstant_d ( cuComplex* f, cuComplex a, int size )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < size )
    {
		f[i].x += a.x;
		f[i].y += a.y;
	}
}


void realPart ( float* Vri, cufftComplex* V, int size )
{
    for ( int i = 0; i < size; i++ )
    { Vri[i] =  V[i].x; }
}

void imagPart ( float* Vri, cufftComplex* V, int size )
{
    for ( int i = 0; i < size; i++ )
    { Vri[i] =  V[i].y; }
}

float sumElements(cuComplex* V, const int size)
{   
	cuComplex dummyThold, sumV;
	dummyThold.x = -FLT_MAX;
	dummyThold.y = -FLT_MAX;

	sumV = sumElements_helper( V, dummyThold, size, 0, 1.f ); // flag is 0 for a regular sum

	return( sumV.x + sumV.y ); 
}


float sumAbsValElements(cuComplex* V, const int size, const float mu0 )
{    
	cuComplex dummyThold, sumV;
	dummyThold.x = -FLT_MAX;
	dummyThold.y = -FLT_MAX;

	sumV = sumElements_helper( V, dummyThold, size, 1, mu0 ); // flag is 1 for a sum of absolute values

	return( sumV.x + sumV.y );
}


cuComplex sumElementsComplex(cuComplex* V, const int size)
{    
	cuComplex dummyThold, sumV;
	dummyThold.x = -FLT_MAX;
	dummyThold.y = -FLT_MAX;

	sumV = sumElements_helper( V, dummyThold, size, 0, 1.f ); // flag is 0 for a regular sum

	return( sumV );
}


cuComplex sumThresElements( cuComplex* V, const int size, cuComplex thold )
{    
	cuComplex sumV;

	sumV = sumElements_helper( V, thold, size, 2, 1.f ); // flag is 2 for a sum of thresholded values

	return( sumV );
}

cuComplex sumThresAbsElements( cuComplex* V, const int size, cuComplex thold )
{    
	cuComplex sumV;

	sumV = sumElements_helper( V, thold, size, 3, 1.f ); // flag is 3 for a sum of thresholded absolute values

	return( sumV );
}


cuComplex sumElements_helper( cuComplex* V, cuComplex thold, const int size, const int flag, const float mu0 )
{
	// flag == 0 for regular sum, 
	// flag == 1 for the sum of the absolute values
	// flag == 2 for the sum of thresholded values
	// flag == 3 for the sum of thresholded absolute values

	const int bS = 128; // Make it high to reduce memory load (Maximum allowed on Tesla K20c: 1024). MUST be a power of 2 (for this application).
	int vecSize = size;
	int sumSize0 = vecSize/bS + 1;
	cuComplex *sum0;
	cuda_assert( cudaMalloc( ( void** ) &sum0, sumSize0 * sizeof( cuComplex ) ) ); 

	sumElements_d <<< sumSize0, bS, bS * sizeof( cuComplex ) >>> ( sum0, V, thold, vecSize, flag, mu0 );
	
	while( sumSize0 > 1 )
	{
		vecSize = sumSize0;
		sumSize0 = sumSize0 / bS + 1; 
		// use the regular sum ( flag == 0 ) here; the operation above has applied the abs val or the threshold already irreversibly to sum0
		sumElements_d <<< sumSize0, bS, bS * sizeof( cuComplex ) >>> ( sum0, sum0, thold, vecSize, 0, 1.f );
	}

	cuComplex sum1;
	cuda_assert( cudaMemcpy( &sum1, sum0, sizeof( cuComplex ), cudaMemcpyDeviceToHost ) );
	cuda_assert( cudaFree( sum0 ) );
	return( sum1 );
}

void cufftShift2D_h( cufftComplex* f0_d, int n1, int n2, int bS )
{
	//First dimension

	cufftComplex* f1_d;
	const int redFctr0 = 64;
	int redFctr = redFctr0;
	
	int n0  = n2 / redFctr;
	if ( n0 < 1 )
	{    n0 = 1; }
	redFctr = n2 / n0;
	int n00 = n2 - n0 * redFctr;

	cuda_assert ( cudaMalloc ( ( void** ) &f1_d, n1 * n0 * sizeof ( cufftComplex ) ) );

	for ( int i = 0; i < redFctr; i++ )
	{
		cufftShift2D_dim1_d <<< ( ( n1 * n0 ) / bS + 1 ), bS >>> ( f1_d, f0_d, n1, n2, n0, n0*i );
		copyCufftShift_dim1_d <<< ( ( n1 * n0 ) / bS + 1 ), bS >>> ( f0_d, f1_d, n1, n2, n0, n0*i );
	}
	cufftShift2D_dim1_d <<< ( ( n1 * n00 ) / bS + 1 ), bS >>> ( f1_d, f0_d, n1, n2, n00, n0*redFctr );
	copyCufftShift_dim1_d <<< ( ( n1 * n00 ) / bS + 1 ), bS >>> ( f0_d, f1_d, n1, n2, n00, n0*redFctr );
	cudaFree( f1_d );

	// Second dimension

	cufftComplex* f2_d;
	redFctr = redFctr0;
	
	n0  = n1 / redFctr;
	if ( n0 < 1 )
	{    n0 = 1; }
	redFctr = n1 / n0;
	n00 = n1 - n0 * redFctr;

	cuda_assert ( cudaMalloc ( ( void** ) &f2_d, n2 * n0 * sizeof ( cufftComplex ) ) );

	for ( int i = 0; i < redFctr; i++ )
	{
		cufftShift2D_dim2_d <<< ( ( n2 * n0 ) / bS + 1 ), bS >>> ( f2_d, f0_d, n1, n2, n0, n0*i );
		copyCufftShift_dim2_d <<< ( ( n2 * n0 ) / bS + 1 ), bS >>> ( f0_d, f2_d, n1, n2, n0, n0*i );
	}
	cufftShift2D_dim2_d <<< ( ( n2 * n00 ) / bS + 1 ), bS >>> ( f2_d, f0_d, n1, n2, n00, n0*redFctr );
	copyCufftShift_dim2_d <<< ( ( n2 * n00 ) / bS + 1 ), bS >>> ( f0_d, f2_d, n1, n2, n00, n0*redFctr );
	
	cudaFree( f2_d );
}

void cufftIShift2D_h( cufftComplex* f0_d, int n1, int n2, int bS )
{
	//First dimension

	cufftComplex* f1_d;
	const int redFctr0 = 64;
	int redFctr = redFctr0;
	
	int n0  = n2 / redFctr;
	if ( n0 < 1 )
	{    n0 = 1; }
	redFctr = n2 / n0;
	int n00 = n2 - n0 * redFctr;

	cuda_assert ( cudaMalloc ( ( void** ) &f1_d, n1 * n0 * sizeof ( cufftComplex ) ) );

	for ( int i = 0; i < redFctr; i++ )
	{
		cufftIShift2D_dim1_d <<< ( ( n1 * n0 ) / bS + 1 ), bS >>> ( f1_d, f0_d, n1, n2, n0, n0*i );
		copyCufftShift_dim1_d <<< ( ( n1 * n0 ) / bS + 1 ), bS >>> ( f0_d, f1_d, n1, n2, n0, n0*i );
	}
	cufftIShift2D_dim1_d <<< ( ( n1 * n00 ) / bS + 1 ), bS >>> ( f1_d, f0_d, n1, n2, n00, n0*redFctr );
	copyCufftShift_dim1_d <<< ( ( n1 * n00 ) / bS + 1 ), bS >>> ( f0_d, f1_d, n1, n2, n00, n0*redFctr );
	cudaFree( f1_d );

	// Second dimension

	cufftComplex* f2_d;
	redFctr = redFctr0;
	
	n0  = n1 / redFctr;
	if ( n0 < 1 )
	{    n0 = 1; }
	redFctr = n1 / n0;
	n00 = n1 - n0 * redFctr;

	cuda_assert ( cudaMalloc ( ( void** ) &f2_d, n2 * n0 * sizeof ( cufftComplex ) ) );

	for ( int i = 0; i < redFctr; i++ )
	{
		cufftIShift2D_dim2_d <<< ( ( n2 * n0 ) / bS + 1 ), bS >>> ( f2_d, f0_d, n1, n2, n0, n0*i );
		copyCufftShift_dim2_d <<< ( ( n2 * n0 ) / bS + 1 ), bS >>> ( f0_d, f2_d, n1, n2, n0, n0*i );
	}
	cufftIShift2D_dim2_d <<< ( ( n2 * n00 ) / bS + 1 ), bS >>> ( f2_d, f0_d, n1, n2, n00, n0*redFctr );
	copyCufftShift_dim2_d <<< ( ( n2 * n00 ) / bS + 1 ), bS >>> ( f0_d, f2_d, n1, n2, n00, n0*redFctr );
	
	cudaFree( f2_d );
}