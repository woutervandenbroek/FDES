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

float * image= NULL;	
float * potential= NULL;
float * exitwave= NULL;



int main (int argc, char **argv)
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
	
	/**********************************************************
	*  parse argv   
	*********************************************************/  

	char input_name[BUZZ_SIZE];
	char image_name[BUZZ_SIZE];
	char emd_name[BUZZ_SIZE];
	bool input_image_name =false;
	bool input_emd_name =  false;
	
	static struct option longOptions[] = {
		{ "gpu_index", 	required_argument, 0, 0 },  		//0
		{ "input_name", required_argument, 0, 0 },  		//1
		{ "image_name", required_argument, 0, 0 },  		//2
		{ "emd_name",   required_argument, 0, 0 },              //3 
		{ "print_level", required_argument, 0, 0},  		//4
		{ "help",	no_argument, 0, 0 },                    //5
		{ "version",	no_argument, 0, 0 },                    //6
		{ NULL, 0, 0, 0 }
		};
		
	while(1) {		
		int optionIndex = 0;
		int c = getopt_long(argc, argv, "", longOptions, &optionIndex);

		if (c == -1) 
		{    break; }  // empty argv
		else if (c == 0 && optionIndex == 0)    //confOption
			gpu_index = atoi(optarg) ;
		else if (c == 0 && optionIndex == 1)	//input_name
		{
			strcpy(input_name, optarg);
			fprintf ( stderr, "  input_name %s  \n", input_name); 
			if ( strstr (input_name, ".emd")!=NULL )
			{
				confOption = 0;
			}
			if ( strstr (input_name, ".cnf")!=NULL  )
			{
				confOption = 1;
			}
			if ( strstr (input_name, ".qsc")!=NULL  )
			{
				confOption = 2;
			}		  
		}
		
		else if (c == 0 && optionIndex == 2)	//image_name
		{
		  strcpy(image_name, optarg);
		  input_image_name = true;
		}
		
		else if (c == 0 && optionIndex == 3)	//emd_name
		{
		  strcpy(emd_name, optarg);
		  input_emd_name = true;
		}
		
		else if (c == 0 && optionIndex == 4)	//printLevel
			printLevel = atoi(optarg) ;

		if (printLevel<0 || printLevel >2)
		{
			fprintf ( stderr, " \n printLevel error %s  \n", input_name);
			exit(EXIT_FAILURE);
		}
		
		else if (c == 0 && optionIndex == 5)     // --help
		{
			fprintf ( stderr, " \nUsage: \n \
			[ --input_name <string specifying the source of simulation parameters(.cnf, .emd or .qsc file)> ] \n\
			[ --emd_name   <string specifying the name of generated emd files>                              ] \n\
			[ --print_level<Flag indicating the level of output >                                          ] \n\
			               <0 output images, default>  \n\
			               <1 output images and potential slices>\n\
			               <2 output images, potential slices and exit waves> \n\
			[ --gpu_index  <Flag indicating the device ID of GPU >                                          ]  \n\
			               <default 0, change it for multiple CUDA-capable GPUs configuration if necessary>\n\
			[ --help       <Flag outputting the usage of FDES >                                             ] \n\
			[ --version    <Flag indicating the version of FDES >                                           ] \n"); 
			exit (EXIT_FAILURE);
		}
		else if (c == 0 && optionIndex == 6)
		{
		  fprintf ( stderr, " \n FDES Version : %1.1f  \n", version); 
		  exit (EXIT_FAILURE);
		}
	}

	timer runTime;
	cuda_assert ( cudaSetDevice ( gpu_index ) );

	if (confOption == -1)
	{
		fprintf ( stderr, "  confOption is not set, using default configuration\n");
		bool ret;
		  
		ret = getParams( default_txt_name, &params, &Z_d, &xyzCoord_d, &DWF_d, &occ_d ); // first read simulation parameters from 'dataFDES.txt'

		if(!ret)
		{
			bool retHdf5;
			retHdf5 = readHdf5 ( default_emd_name,  &params, &Z_d, &xyzCoord_d, &DWF_d, &occ_d);// then try to read simulation parameters from 'config.emd'
			if(!retHdf5)
			{	fprintf ( stderr, "No valid configuration for simulation!.\n" );
			exit (EXIT_FAILURE);
			}
		}
		else
		{
			confOption = 1;
		}		  

	}
	else
	{
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
			ret =getParams(input_name, &params, &Z_d, &xyzCoord_d, &DWF_d, &occ_d );
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
	}

	printLevelOptimization( params );

	if(confOption!=0)
	{
		writeHdf5 ( default_emd_name,  params,  &Z_d, &xyzCoord_d, &DWF_d, &occ_d);
	}

	if(!input_image_name )
	{
		strcpy(image_name,default_image_name);
	}

	if(!input_emd_name )
	{
		strcpy(emd_name,default_emd_save_name);
	}


	/**********************************************************
	simulation kernel
	*********************************************************/  		

	buildMeasurements( params, Z_d, xyzCoord_d, DWF_d, occ_d,  image_name, emd_name);
	
	/**********************************************************
        free Memory
	*********************************************************/  			
	
	freeParams ( &params );
	cuda_assert ( cudaFree ( xyzCoord_d ) );
	cuda_assert ( cudaFree ( DWF_d ) );
	cuda_assert ( cudaFree ( occ_d ) );	
	cuda_assert ( cudaFree ( Z_d ) );	
	
	fprintf ( stderr, "  Done.\n" );
	return ( 0 );
}


void showLicense()
{
	fprintf(stderr, "\n  FDES 1.0  Copyright (C) 2015  Wouter Van den Broek, Xiaoming Jiang\n");
	fprintf(stderr,   "  This program comes with ABSOLUTELY NO WARRANTY,\n");
	fprintf(stderr,   "  this is free software, and you are welcome to redistribute\n");
	fprintf(stderr,   "  it under certain conditions; for details\n");
	fprintf(stderr,   "  see LICENSE.txt or http://www.gnu.org/licenses/.\n\n");
}

