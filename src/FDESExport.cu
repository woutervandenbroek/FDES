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

#include "FDES.h"
const char*   default_txt_name  =    "dataFDES.cnf";
const char*   default_emd_name  =    "config.emd";
const char*   default_qsc_name  =    "test.qsc";
const char*   default_image_name =   "Measurements.bin";
const char*   default_emd_save_name ="results.emd";

int MATLAB_TILT_COMPATIBILITY = 0;
int NO_FUNCTION_EVALS_FDES = 0;
int NO_DERIVATIVE_EVALS_FDES = 0;
int printLevel = 0; //Flag indicating the level of output files  0 output images  1 output images and  potential slice 2 output are together with exit waves 
int confOption = -1; // variable to control source of simulation parameters.    0 from .emd file; 1 from .txt file; 2 from .qsc file 
int gpu_index = 0;
float version = 0.1;
bool atomsFromExternal = false;
float * image = NULL;	
float * potential = NULL;
float * exitwave = NULL;


extern "C" {__declspec(dllexport)
void FDES (int gpu_Index, int print_Level, char * input_name, char * image_name, char * emd_save_name, float * atomsArray, int numAtoms, float * dstImage)
{
	/**********************************************************
	*  show License of FDES
	*********************************************************/  
 	showLicense ();
	
	params_t* params = NULL;
	float *	xyzCoord_d= NULL; 
	float *	DWF_d= NULL;
	float *	occ_d= NULL;
	int   *	Z_d= NULL;
	
	atomsFromExternal = true;
 
// 	char image_name[BUZZ_SIZE];
// 	char emd_name[BUZZ_SIZE];
	bool input_image_name =false;
	bool input_emd_name =  false;
	
	gpu_index = gpu_Index;
	
	printLevel =print_Level;

	fprintf ( stderr, "   input_name %s  \n", input_name); 
	
	if ( strstr (input_name, ".emd")!=NULL )
	{
		confOption = 0;
	}
	else if ( strstr (input_name, ".cnf")!=NULL  )
	{
		confOption = 1;
	} 
	else if ( strstr (input_name, ".qsc")!=NULL  )
	{
		confOption = 2;
	}
	else
	{
		fprintf ( stderr, " \n input file %s error   \n", input_name); 
		exit(0);
	}
	  

	timer runTime;
	cuda_assert ( cudaSetDevice ( gpu_index ) );

	bool ret;                           //confOption is set through argv
	
	switch (confOption)
		{
		case 0:

			ret = readHdf5 ( input_name,  &params, &Z_d, &xyzCoord_d, &DWF_d, &occ_d);
			if(!ret)
			{
				fprintf ( stderr, " \n Errors occur when reading \"config.emd\" \n");
				exit (EXIT_FAILURE);
			}
			break;

		case 1:
			ret = getParams(input_name, &params, &Z_d, &xyzCoord_d, &DWF_d, &occ_d );
			if(!ret)
			{
				fprintf ( stderr, " \n Errors occur when reading \"dataFDES.txt\" \n");
				exit (EXIT_FAILURE);
			}

			writeHdf5 ( default_emd_name,  params,  &Z_d, &xyzCoord_d, &DWF_d, &occ_d);
			break;

		case 2: 

			ret = readQsc(input_name, &params, &Z_d, &xyzCoord_d, &DWF_d, &occ_d );
			// 		      exit(EXIT_FAILURE);

			break;
		default:
			exit (EXIT_FAILURE);
		}	
		

	readAtomsFromArray(params, &Z_d, &xyzCoord_d, &DWF_d, &occ_d , atomsArray, numAtoms);
	fprintf(stderr,"  Number of atoms %d \n", (*params).SAMPLE.nAt);
// 
	printLevelOptimization( params );
// 
	if(confOption!=0)
	{
		writeHdf5 ( default_emd_name,  params,  &Z_d, &xyzCoord_d, &DWF_d, &occ_d);
	}
// 


	/**********************************************************
	simulation kernel
	*********************************************************/  		

	buildMeasurements( params, Z_d, xyzCoord_d, DWF_d, occ_d,  image_name, emd_save_name);
	
 	exportFormedimage(params, dstImage, image );
	
	/**********************************************************
        free Memory
	*********************************************************/  	
	
	
	freeParams ( &params );
	cuda_assert ( cudaFree ( xyzCoord_d ) );
	cuda_assert ( cudaFree ( DWF_d ) );
	cuda_assert ( cudaFree ( occ_d ) );	
	cuda_assert ( cudaFree ( Z_d ) );	
	
// 	fprintf ( stderr, "  Done.\n" );
// 	return ( 0 );
}
}


void showLicense()
{
	fprintf(stderr, "\n  FDES 1.0  Copyright (C) 2015  Wouter Van den Broek, Xiaoming Jiang\n");
	fprintf(stderr,   "  This program comes with ABSOLUTELY NO WARRANTY,\n");
	fprintf(stderr,   "  this is free software, and you are welcome to redistribute\n");
	fprintf(stderr,   "  it under certain conditions; for details\n");
	fprintf(stderr,   "  see LICENSE.txt or http://www.gnu.org/licenses/.\n\n");
}

