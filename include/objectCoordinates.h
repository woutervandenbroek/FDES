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