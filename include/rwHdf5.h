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

#ifndef RWHDF5
#define RWHDF5
#include <stdlib.h>
#include <stdio.h>
#include "hdf5.h"
#include"paramStructure.h"
#include <string.h>
#include <typeinfo>   // operator typeid


#ifdef __linux__
#include <stdbool.h>
#endif
 
 
#include <malloc.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include "cuda_assert.hpp"


void writeHdf5 (const char* filename,  params_t* params,int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d);

void writeHdf5 (const char* filename, float* f,float * pPotential, params_t* params);

void writeHdf5 (const char* filename, float* image,float * potential, float * exitwave, params_t* params,int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d);

bool readHdf5 ( const char* filename,  params_t** params,int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d);

template < typename  Type> void writeAttributeSingle(hid_t loc_id, const char * attrName,  Type value);
void writeAttributeString(hid_t loc_id, const char * attrName,  const char*  attrString);

bool readAttributeString(hid_t loc_id, const char * attrName,   char*  attrString);

template < typename  Type> bool readAttributeSingle (hid_t loc_id, const char * attrName,  Type * value);


extern int printLevel;

extern float version;

extern bool atomsFromExternal;

#endif