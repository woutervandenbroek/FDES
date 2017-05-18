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

#include "objectCoordinates.h"
# include "paramStructure.h"

void getCoordinates( int nAt, int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d )
{
	FILE* fr;
	fr = fopen ( "Params.cnf", "rt" );

	cuda_assert ( cudaMalloc ( ( void** ) Z_d, nAt * sizeof ( int ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) xyzCoord_d, nAt * 3 * sizeof ( float ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) DWF_d, nAt * sizeof ( float ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) occ_d, nAt * sizeof ( float ) ) );

	int *Z_h;
	float *xyzCoord_h, *DWF_h, *occ_h;
	Z_h = ( int* ) malloc ( nAt * sizeof ( int ) );
	xyzCoord_h = ( float* ) malloc ( nAt * 3 * sizeof ( float ) );
	DWF_h = ( float* ) malloc ( nAt * sizeof ( float ) );
	occ_h = ( float* ) malloc ( nAt * sizeof ( float ) );

	// Read values
	readCoordinates ( fr, Z_h, xyzCoord_h, DWF_h, occ_h );

	cuda_assert ( cudaMemcpy ( *Z_d, Z_h,  nAt * sizeof ( int ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( *xyzCoord_d, xyzCoord_h, nAt * 3 * sizeof ( int ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( *DWF_d, DWF_h, nAt * sizeof ( float ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( *occ_d, occ_h, nAt * sizeof ( float ), cudaMemcpyHostToDevice ) );

	fclose ( fr );
	free( Z_h );
	free( xyzCoord_h );
	free( DWF_h );
	free( occ_h );
}


void freeCoordinatesVars ( int* Z_d, float* xyzCoord_d, float* DWF_d, float* occ_d )
{
	cuda_assert ( cudaFree ( Z_d ) );
	cuda_assert ( cudaFree ( xyzCoord_d ) );
	cuda_assert ( cudaFree ( DWF_d ) );
	cuda_assert ( cudaFree ( occ_d ) );
}


void readCoordinates ( FILE* fr, int* Z_h, float* xyzCoord_h, float* DWF_h, float* occ_h )
{
	fseek ( fr, 0, SEEK_SET );
	const int size = 200;
	char* line;
	line = ( char* ) malloc ( size * sizeof ( char ) );
	char* fieldName;
	fieldName = ( char* ) malloc ( size * sizeof ( char ) );
	int j = 0;

	while ( !feof( fr ) ) {
		fgets ( line, size, fr );
		sscanf ( line, "%s", fieldName );
		if ( !strncmp ( fieldName, "atom:", 5 ) ) {    
			sscanf ( line, "%*s %i %g %g %g %g %g", & Z_h[j], & xyzCoord_h[3 * j + 0], & xyzCoord_h[3 * j + 1], & xyzCoord_h[3 * j + 2], & DWF_h[j], & occ_h[j] );
			j += 1;
		}
		resetLine ( line );
	}

	free ( line );
	free ( fieldName );
}

void resetLine( char* line )
{
	line[0] = *( "#" ); // Reset "loremipsum" to "#oremipsum", indicating a comment.
}