/*------------------------------------------------
	Copyright (C) 2013 Wouter Van den Broek
	See IDES.cu for full notice
------------------------------------------------*/

#ifndef objectCoordinates_ouhasfgozbljhgouzgsadfrjkhabvwrktzg
#define objectCoordinates_ouhasfgozbljhgouzgsadfrjkhabvwrktzg

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include "cuda_assert.hpp"

void getCoordinates( int nAt, int** Z, float** xyzCoord, float** DWF, float** occ );

void freeCoordinatesVars ( int* Z_d, float* xyzCoord_d, float* DWF_d, float* occ_d );

void readCoordinates ( FILE* fr, int* Z_h, float* xyzCoord_h, float* DWF_h, float* occ_h );

void resetLine( char* line );

#endif 