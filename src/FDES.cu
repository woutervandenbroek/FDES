/*==================================================================

FDES, forward dynamical electron scattering, 
is a GPU-based multislice algorithm.

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

Associated article: 
W. Van den Broek, X. Jiang, C.T. Koch. FDES, a GPU-based multislice 
algorithm with increased effciency of the computation of the 
projected potential. Ultramicroscopy (2015).

Address: Institut for Experimentel Physics
         Ulm University
         Albert-Einstein-Allee 11
         89081 Ulm
         Germany

===================================================================*/


#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <float.h>
#include <math.h>
#include <cufft.h>
#include "paramStructure.h"
#include "rwBinary.h"
#include "optimFunctions.h"
#include "multisliceSimulation.h"
#include "cuda_assert.hpp"
#include "complexMath.h"
#include "coordArithmetic.h"
#include "performanceTimer.h"
#include "projectedPotential.h"
#include "optimFunctions.h"
#include "objectCoordinates.h"
#include "crystalMaker.h"
#include "globalVariables.h"

void showLicense ();

int MATLAB_TILT_COMPATIBILITY = 0;
int NO_FUNCTION_EVALS_IDES = 0;
int NO_DERIVATIVE_EVALS_IDES = 0;
int gpu_index = 0;

int main ()
{
	showLicense ();

	fprintf ( stderr, "\n  Starting FDES.\n" );

	if ( true ) { // Always-true if-statement for runTime
		timer runTime;

		gpu_index = readCudaDeviceRank( );
		cuda_assert ( cudaSetDevice ( gpu_index ) );

		float *xyzCoord_d, *DWF_d, *occ_d;
		int *Z_d;
		params_t* params;

		getParams( &params );

		getCoordinates( params->SAMPLE.nAt, &Z_d, &xyzCoord_d, &DWF_d, &occ_d );

		writeConfig ( "ParamsUsed.cnf", params, Z_d, xyzCoord_d, DWF_d, occ_d  );

		buildMeasurements( params, Z_d, xyzCoord_d, DWF_d, occ_d );
		//buildMeasurementsTimer( params, Z_d, xyzCoord_d, DWF_d, occ_d );
		
		freeCoordinatesVars ( Z_d, xyzCoord_d, DWF_d, occ_d );
		freeParams ( &params );
	}
	
	fprintf ( stderr, "  Done.\n" );
	return ( 0 );
}


void showLicense()
{
	fprintf(stderr, "\n  FDES 1.0  Copyright (C) 2013  Wouter Van den Broek\n");
	fprintf(stderr,   "  This program comes with ABSOLUTELY NO WARRANTY,\n");
	fprintf(stderr,   "  this is free software, and you are welcome to redistribute\n");
	fprintf(stderr,   "  it under certain conditions; for details\n");
	fprintf(stderr,   "  see License_IDES.txt or http://www.gnu.org/licenses/.\n");
}
