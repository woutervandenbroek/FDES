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

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include "paramStructure.h"
#include "cuda_assert.hpp"
#include "cufft_assert.h"
#include "cublas_assert.hpp"
#include "projectedPotential.h"
#include "optimFunctions.h"
#include "globalVariables.h"
#include "hdf5.h"



bool readConfig ( const char* file, params_t* params,  int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d  )
{
    bool ret = true;
    FILE* fr;
    const int size = 100;
    char* line;
    line = ( char* ) malloc ( size * sizeof ( char ) );
    char* fieldName;
    fieldName = ( char* ) malloc ( size * sizeof ( char ) );
    int ts_i = 0;
    int tb_i = 0;
    int df_i = 0;

    fr = fopen ( file, "rt" );
    
    if (fr==NULL)
    {
      fprintf ( stderr, "\n  Not able to read simulation configuration from %s, try to read from 'config.emd' \n", file );
      ret = false;
      return ret;
    }

	do
	{
		if (fgets ( line, size, fr )!= NULL)
		sscanf ( line, "%s", fieldName );

		if ( !strncmp ( fieldName, "voltage:", 8 ) )
		{    sscanf ( line, "%*s %g", & ( *params ).EM.E0 ); }


		if ( !strncmp ( fieldName, "C1:", 3 ) )
		{	sscanf ( line, "%*s %g ", & ( *params ).EM.aberration.C1_0); }

		if ( !strncmp ( fieldName, "A1:", 3 ) )
		{	sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.A1_0, & ( *params ).EM.aberration.A1_1 ); }

		if ( !strncmp ( fieldName, "A2:", 3 ) )
		{	sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.A2_0, & ( *params ).EM.aberration.A2_1 ); }

		if ( !strncmp ( fieldName, "B2:", 3 ) )
		{	sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.B2_0, & ( *params ).EM.aberration.B2_1 ); }

		if ( !strncmp ( fieldName, "C3:", 3 ) )
		{	sscanf ( line, "%*s %g", & ( *params ).EM.aberration.C3_0); }

		if ( !strncmp ( fieldName, "A3:", 3 ) )
		{    sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.A3_0, & ( *params ).EM.aberration.A3_1 );	}

		if ( !strncmp ( fieldName, "S3:", 3 ) )
		{	sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.S3_0, & ( *params ).EM.aberration.S3_1 ); }

		if ( !strncmp ( fieldName, "A4:", 3 ) )
		{	sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.A4_0, & ( *params ).EM.aberration.A4_1 ); }

		if ( !strncmp ( fieldName, "B4:", 3 ) )
		{	sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.B4_0, & ( *params ).EM.aberration.B4_1 ); }

		if ( !strncmp ( fieldName, "D4:", 3 ) )
		{	sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.D4_0, & ( *params ).EM.aberration.D4_1 ); }

		if ( !strncmp ( fieldName, "C5:", 3 ) )
		{	sscanf ( line, "%*s %g", & ( *params ).EM.aberration.C5_0); }

		if ( !strncmp ( fieldName, "A5:", 3 ) )
		{	sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.A5_0, & ( *params ).EM.aberration.A5_1 ); }

		if ( !strncmp ( fieldName, "R5:", 3 ) )
		{	sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.R5_0, & ( *params ).EM.aberration.R5_1 ); }

		if ( !strncmp ( fieldName, "S5:", 3 ) )
		{	sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.S5_0, & ( *params ).EM.aberration.S5_1 ); }

		if ( !strncmp ( fieldName, "focus_spread:", 12 ) )
		{	sscanf ( line, "%*s %g", & ( *params ).EM.defocspread ); }

		if ( !strncmp ( fieldName, "illumination_angle:", 19 ) )
		{	sscanf ( line, "%*s %g", & ( *params ).EM.illangle ); }

		if ( !strncmp ( fieldName, "mtf_a:", 6 ) )
		{	sscanf ( line, "%*s %g", & ( *params ).EM.mtfa ); }

		if ( !strncmp ( fieldName, "mtf_b:", 6 ) )
		{	sscanf ( line, "%*s %g", & ( *params ).EM.mtfb ); }

		if ( !strncmp ( fieldName, "mtf_c:", 6 ) )
		{	sscanf ( line, "%*s %g", & ( *params ).EM.mtfc ); }

		if ( !strncmp ( fieldName, "mtf_d:", 6 ) )
		{    sscanf ( line, "%*s %g", & ( *params ).EM.mtfd ); }

		if ( !strncmp ( fieldName, "objective_aperture:", 19 ) )
		{	sscanf ( line, "%*s %g", & ( *params ).EM.ObjAp ); }

		if ( !strncmp ( fieldName, "sample_size_x:", 14 ) )
		{	sscanf ( line, "%*s %i", & ( *params ).IM.m1 ); }

		if ( !strncmp ( fieldName, "sample_size_y:", 14 ) )
		{	sscanf ( line, "%*s %i", & ( *params ).IM.m2 );	}

		if ( !strncmp ( fieldName, "sample_size_z:", 14 ) )
		{    sscanf ( line, "%*s %i", & ( *params ).IM.m3 ); }

		if ( !strncmp ( fieldName, "pixel_size_x:", 13 ) )
		{	sscanf ( line, "%*s %g", & ( *params ).IM.d1 ); }

		if ( !strncmp ( fieldName, "pixel_size_y:", 13 ) )
		{	sscanf ( line, "%*s %g", & ( *params ).IM.d2 ); }

		if ( !strncmp ( fieldName, "pixel_size_z:", 13 ) )
		{	sscanf ( line, "%*s %g", & ( *params ).IM.d3 ); }

		if ( !strncmp ( fieldName, "border_size_x:", 14 ) )
		{	sscanf ( line, "%*s %i", & ( *params ).IM.dn1 ); }

		if ( !strncmp ( fieldName, "border_size_y:", 14 ) )
		{	sscanf ( line, "%*s %i", & ( *params ).IM.dn2 ); }

		if ( !strncmp ( fieldName, "image_size_x:", 13 ) )
		{	sscanf ( line, "%*s %i", & ( *params ).IM.n1 ); }

		if ( !strncmp ( fieldName, "image_size_y:", 13 ) )
		{	sscanf ( line, "%*s %i", & ( *params ).IM.n2 ); }

		if ( !strncmp ( fieldName, "image_size_z:", 13 ) )
		{	sscanf ( line, "%*s %i", & ( *params ).IM.n3 ); }

		if ( !strncmp ( fieldName, "specimen_tilt:", 14) )
		{	
			sscanf ( line, "%*s %g %g", & ( *params ).IM.tiltspec[2 * ts_i], & ( *params ).IM.tiltspec[2 * ts_i + 1] );
			ts_i += 1;
		}

		if ( !strncmp ( fieldName, "beam_tilt:", 10 ) )
		{
			sscanf ( line, "%*s %g %g", & ( *params ).IM.tiltbeam[2 * tb_i], & ( *params ).IM.tiltbeam[2 * tb_i + 1] );
			tb_i += 1;
		}

		if ( !strncmp ( fieldName, "defoci:", 7 ) )
		{
			sscanf ( line, "%*s %g", & ( *params ).IM.defoci[df_i] );
			df_i += 1;
		}
		
		if ( !strncmp ( fieldName, "user_name:", 11) )
		{
			sscanf ( line, "user_name: %[^\n]s\n", ( *params ).USER.user_name );
		}
		
		if ( !strncmp ( fieldName, "institution:", 12) )
		{
			sscanf ( line, "institution: %[^\n]s\n", ( *params ).USER.institution );
		}
				
		if ( !strncmp ( fieldName, "department:", 11) )
		{
			sscanf ( line, "department: %[^\n]s\n", ( *params ).USER.department );
		}
		
		if ( !strncmp ( fieldName, "email:", 6) )
		{
			sscanf ( line, "email: %[^\n]s\n", ( *params ).USER.email );
		}
		
		if ( !strncmp ( fieldName, "comment:", 8) )
		{
			sscanf ( line, "comment: %[^\n]s\n", ( *params).COMMENT.comments );
		}
		
	        if ( !strncmp ( fieldName, "sample_name:", 12) )
		{
			sscanf ( line, "sample_name: %[^\n]s\n", ( *params).SAMPLE.sample_name);

		}

		if ( !strncmp ( fieldName, "material:", 9) )
		{
			sscanf ( line, "material: %[^\n]s\n",(*params).SAMPLE.material );
		}

		if ( !strncmp ( fieldName, "absorptive_potential_factor:", 28 ))
		{
			sscanf ( line, "%*s %g", &(*params).SAMPLE.imPot);
		}							

	        if ( !strncmp ( fieldName, "pixel_dose:", 11 ) )
		{
			sscanf ( line, "%*s %g", &(*params).IM.pD);
		}
		
		if ( !strncmp ( fieldName, "frozen_phonons:", 15 ) )
		{
			sscanf ( line, "%*s %d", &(*params).IM.frPh);
		}	
		
		if ( !strncmp ( fieldName, "subpixel_size_z:", 16 ) )
		{
			sscanf ( line, "%*s %g", &(*params).IM.subSlTh);
		}
		
		if ( !strncmp ( fieldName, "specimen_tilt_offset_x:", 23 ) )
		{
			sscanf ( line, "%*s %g", &(*params).IM.specimen_tilt_offset_x);
		}
		
		if ( !strncmp ( fieldName, "specimen_tilt_offset_y:", 23 ) )
		{
			sscanf ( line, "%*s %g", &(*params).IM.specimen_tilt_offset_y);
		}
				
		if ( !strncmp ( fieldName, "specimen_tilt_offset_z:", 23 ) )
		{
			sscanf ( line, "%*s %g", &(*params).IM.specimen_tilt_offset_z);
		}
		
		
		if ( !strncmp ( fieldName, "mode:", 5 ) )
		{
			sscanf ( line, "%*s %d", &(*params).IM.mode);
		}
		
		
	}
	while ( !feof( fr ) );
	
	if(!atomsFromExternal)
	{
		params->SAMPLE.nAt= numberOfAtoms( fr );

		fprintf (stderr, "  Number of atoms in the specimen: %i \n ", (*params).SAMPLE.nAt);

		cuda_assert ( cudaMalloc ( ( void** ) Z_d,       (*params).SAMPLE.nAt* sizeof ( int ) ) );
		cuda_assert ( cudaMalloc ( ( void** ) xyzCoord_d,(*params).SAMPLE.nAt * 3 * sizeof ( float ) ) );
		cuda_assert ( cudaMalloc ( ( void** ) DWF_d,     (*params).SAMPLE.nAt * sizeof ( float ) ) );
		cuda_assert ( cudaMalloc ( ( void** ) occ_d,     (*params).SAMPLE.nAt * sizeof ( float ) ) );
		
		int *Z_h;
		float *xyzCoord_h, *DWF_h, *occ_h;
		Z_h = ( int* ) malloc ( (*params).SAMPLE.nAt * sizeof ( int ) );
		xyzCoord_h = ( float* ) malloc ( (*params).SAMPLE.nAt * 3 * sizeof ( float ) );
		DWF_h = ( float* ) malloc ( (*params).SAMPLE.nAt * sizeof ( float ) );
		occ_h = ( float* ) malloc ( (*params).SAMPLE.nAt * sizeof ( float ) );

		readCoordinates ( fr, Z_h , xyzCoord_h, DWF_h, occ_h  );

		cuda_assert ( cudaMemcpy ( *Z_d,       Z_h ,       (*params).SAMPLE.nAt*sizeof ( int ), cudaMemcpyHostToDevice ) );
		cuda_assert ( cudaMemcpy ( *xyzCoord_d,xyzCoord_h, (*params).SAMPLE.nAt* 3 * sizeof ( float ), cudaMemcpyHostToDevice ) );
		cuda_assert ( cudaMemcpy ( *DWF_d,     DWF_h,      (*params).SAMPLE.nAt *sizeof ( float ), cudaMemcpyHostToDevice ) );
		cuda_assert ( cudaMemcpy ( *occ_d,     occ_h,      (*params).SAMPLE.nAt* sizeof ( float ), cudaMemcpyHostToDevice ) );

		free(Z_h);
		free(xyzCoord_h);
		free(DWF_h);
		free(occ_h);
	}
	fclose ( fr );
	free ( line );
	free ( fieldName );
	return ret;
}

void readAtomsFromArray(params_t* params, int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d , float * atomsArray, int numAtoms )
{
      if(numAtoms==0||numAtoms<0)
      {		exit(0);   }
      (*params).SAMPLE.nAt=numAtoms;
      int *Z_h;
      float *xyzCoord_h, *DWF_h, *occ_h;
      Z_h = ( int* ) malloc ( (*params).SAMPLE.nAt * sizeof ( int ) );
      xyzCoord_h = ( float* ) malloc ( (*params).SAMPLE.nAt * 3 * sizeof ( float ) );
      DWF_h = ( float* ) malloc ( (*params).SAMPLE.nAt * sizeof ( float ) );
      occ_h = ( float* ) malloc ( (*params).SAMPLE.nAt * sizeof ( float ) );
      
      for (int i=0; i<numAtoms; i++ )
      {
	Z_h[i] = (int)atomsArray[6*i+0];
        xyzCoord_h[3*i+0] = atomsArray[6*i+1];
	xyzCoord_h[3*i+1] = atomsArray[6*i+2];
	xyzCoord_h[3*i+2] = atomsArray[6*i+3];
	DWF_h[i] = atomsArray[6*i+4];
	occ_h[i] = (int)atomsArray[6*i+5];
      }
      
      
      cuda_assert ( cudaMalloc ( ( void** ) Z_d,       (*params).SAMPLE.nAt* sizeof ( int ) ) );
      cuda_assert ( cudaMalloc ( ( void** ) xyzCoord_d,(*params).SAMPLE.nAt * 3 * sizeof ( float ) ) );
      cuda_assert ( cudaMalloc ( ( void** ) DWF_d,     (*params).SAMPLE.nAt * sizeof ( float ) ) );
      cuda_assert ( cudaMalloc ( ( void** ) occ_d,     (*params).SAMPLE.nAt * sizeof ( float ) ) );
      
      
      
      cuda_assert ( cudaMemcpy ( *Z_d,       Z_h ,       (*params).SAMPLE.nAt*sizeof ( int ), cudaMemcpyHostToDevice ) );
      cuda_assert ( cudaMemcpy ( *xyzCoord_d,xyzCoord_h, (*params).SAMPLE.nAt* 3 * sizeof ( float ), cudaMemcpyHostToDevice ) );
      cuda_assert ( cudaMemcpy ( *DWF_d,     DWF_h,      (*params).SAMPLE.nAt *sizeof ( float ), cudaMemcpyHostToDevice ) );
      cuda_assert ( cudaMemcpy ( *occ_d,     occ_h,      (*params).SAMPLE.nAt* sizeof ( float ), cudaMemcpyHostToDevice ) );

      free(Z_h);
      free(xyzCoord_h);
      free(DWF_h);
      free(occ_h);
      
       writeConfig("testRead.txt", params, Z_d,  xyzCoord_d, DWF_d, occ_d );
}

void exportFormedimage(params_t* params, float * dstImage, float * srcImage )
{
  int n1= (*params).IM.n1;
  int n2= (*params).IM.n2;
  int n3= (*params).IM.n3;
  
  int n123 = n1*n2*n3;
  for(int i=0;i< n123;i++)
  {
    dstImage[i] =srcImage[i];
  }
  
}


void writeConfig ( const char* file, params_t* params, int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d)
{
    int *Z_h;
    float *xyzCoord_h, *DWF_h, *occ_h;
    int nAt = params->SAMPLE.nAt;
    
    Z_h = ( int* ) malloc ( nAt * sizeof ( int ) );
    xyzCoord_h = ( float* ) malloc ( nAt * 3 * sizeof ( float ) );
    DWF_h = ( float* ) malloc ( nAt * sizeof ( float ) );
    occ_h = ( float* ) malloc ( nAt * sizeof ( float ) );
    
    cuda_assert ( cudaMemcpy ( Z_h, *Z_d, nAt * sizeof ( int ), cudaMemcpyDeviceToHost) );
    cuda_assert ( cudaMemcpy ( xyzCoord_h, *xyzCoord_d, nAt * 3 * sizeof ( float ), cudaMemcpyDeviceToHost ) );
    cuda_assert ( cudaMemcpy ( DWF_h, *DWF_d, nAt * sizeof ( float ), cudaMemcpyDeviceToHost ) );
    cuda_assert ( cudaMemcpy ( occ_h, *occ_d, nAt * sizeof ( float ), cudaMemcpyDeviceToHost ) );

    
    
    FILE* fw;
    fw = fopen ( file, "wt" );

    fprintf ( fw, "# Parameter file. Let Comments be preceded by '#'\n" );
	fprintf ( fw, "\n# Nature's constants\n#-------------------\n\n" );
    fprintf ( fw, "%s %14.8g %s\n", "m0: ", params->cst.m0, " # Electron rest mass [kg]" );
    fprintf ( fw, "%s %14.8g %s\n", "c: ", params->cst.c, " # Speed of light [m/s]" );
    fprintf ( fw, "%s %14.8g %s\n", "e: ", params->cst.e, " # Elementary charge [C]" );
    fprintf ( fw, "%s %14.8g %s\n", "h: ", params->cst.h, " # Planck's constant [Js]" );
    fprintf ( fw, "%s %14.8g %s\n", "pi: ", params->cst.pi, " # Pi [dimensionless]" );
	
	fprintf ( fw, "\n# Microscope parameters\n#----------------------\n\n" );
    fprintf ( fw, "%s %14.8g %s\n", "voltage: ", params->EM.E0, " # Acceleration voltage [V]" );
    fprintf ( fw, "%s %14.8g %s\n", "gamma: ", params->EM.gamma, " # From relativity: 1+e*E0/m0/c^2" );
    fprintf ( fw, "%s %14.8g %s\n", "lambda: ", params->EM.lambda, " # Electron wavelength [m]" );
    fprintf ( fw, "%s %14.8g %s\n", "sigma: ", params->EM.sigma, " # Interaction constant [1/(Vm)]" );

	fprintf ( fw, "\n%s %14.8g %s\n", "focus_spread: ", params->EM.defocspread, " # Defocus spread for the temporal partial coherence [m]" );
    fprintf ( fw, "%s %14.8g %s\n", "illumination_angle: ", params->EM.illangle, " # illumination half angle characterizing the spatial partial coherence [rad]" );
    fprintf ( fw, "%s %14.8g %s\n", "mtf_a: ", params->EM.mtfa, " # MTF parameters, see Microsc. Microanal. 18 (2012) 336–342." );
    fprintf ( fw, "%s %14.8g\n", "mtf_b: ", params->EM.mtfb );
    fprintf ( fw, "%s %14.8g\n", "mtf_c: ", params->EM.mtfc );
    fprintf ( fw, "%s %14.8g\n", "mtf_d: ", params->EM.mtfd );
    fprintf ( fw, "%s %14.8g %s\n", "objective_aperture: ", params->EM.ObjAp, " # Radius of the objective aperture [rad]" );


	fprintf ( fw, "\n# aberration coefficients\n" );

    fprintf ( fw, "%s %14.8g %14.8g %s\n", "C1: ", params->EM.aberration.C1_0, params->EM.aberration.C1_1, " # Focus value [m]" );
    fprintf ( fw, "%s %14.8g %14.8g %s\n", "A1: ", params->EM.aberration.A1_0, params->EM.aberration.A1_1, " # Astigmatism [m] and [rad]" );
    fprintf ( fw, "%s %14.8g %14.8g\n", "A2: ", params->EM.aberration.A2_0, params->EM.aberration.A2_1 );
    fprintf ( fw, "%s %14.8g %14.8g\n", "B2: ", params->EM.aberration.B2_0, params->EM.aberration.B2_1 );
    fprintf ( fw, "%s %14.8g %14.8g %s\n", "C3: ", params->EM.aberration.C3_0, params->EM.aberration.C3_1, " # Spherical aberration [m]" );
    fprintf ( fw, "%s %14.8g %14.8g\n", "A3: ", params->EM.aberration.A3_0, params->EM.aberration.A3_1 );
    fprintf ( fw, "%s %14.8g %14.8g\n", "S3: ", params->EM.aberration.S3_0, params->EM.aberration.S3_1 );
    fprintf ( fw, "%s %14.8g %14.8g\n", "A4: ", params->EM.aberration.A4_0, params->EM.aberration.A4_1 );
    fprintf ( fw, "%s %14.8g %14.8g\n", "B4: ", params->EM.aberration.B4_0, params->EM.aberration.B4_1 );
    fprintf ( fw, "%s %14.8g %14.8g\n", "D4: ", params->EM.aberration.D4_0, params->EM.aberration.D4_1 );
    fprintf ( fw, "%s %14.8g %14.8g\n", "C5: ", params->EM.aberration.C5_0, params->EM.aberration.C5_1 );
    fprintf ( fw, "%s %14.8g %14.8g\n", "A5: ", params->EM.aberration.A5_0, params->EM.aberration.A5_1 );
    fprintf ( fw, "%s %14.8g %14.8g\n", "R5: ", params->EM.aberration.R5_0, params->EM.aberration.R5_1 );
    fprintf ( fw, "%s %14.8g %14.8g\n", "S5: ", params->EM.aberration.S5_0, params->EM.aberration.S5_1 );


	fprintf ( fw, "\n# Imaging parameters\n#-------------------\n\n" );

	fprintf ( fw, "%s %i %s\n", "mode: ", params->IM.mode, " # 0, 1 or 2 for imaging, diffraction or CBED, resp." );
	fprintf ( fw, "%s %i %s\n", "gpu_index: ", gpu_index, " # Number of the device the code runs on, it's often 0 or 1" );
    fprintf ( fw, "%s %i %s\n", "sample_size_x: ", params->IM.m1, " # Width of the object: length of 2nd dimension or number of columns" );
    fprintf ( fw, "%s %i %s\n", "sample_size_y: ", params->IM.m2, " # Height of the object: length of 1st dimension or number rows" );
    fprintf ( fw, "%s %i %s\n", "sample_size_z: ", params->IM.m3, " # Depth of the object: number of sample_size_y by sample_size_x slices" );
    fprintf ( fw, "%s %14.8g %s\n", "pixel_size_x: ", params->IM.d1, " # Length of 2nd dimension of the voxels [m]" );
    fprintf ( fw, "%s %14.8g %s\n", "pixel_size_y: ", params->IM.d2, " # Length of 1st dimension of the voxels [m]" );
    fprintf ( fw, "%s %14.8g %s\n", "pixel_size_z: ", params->IM.d3, " # Length of 3rd dimension of the voxels [m] of the saved potential_slices" );
    fprintf ( fw, "%s %i %s\n", "border_size_x: ", params->IM.dn1, " # Padding, sample_size_x = image_size_x + 2*border_size_x" );
    fprintf ( fw, "%s %i %s\n", "border_size_y: ", params->IM.dn2, " # Padding, sample_size_y = image_size_y + 2*border_size_y" );
    fprintf ( fw, "%s %i %s\n", "image_size_x: ", params->IM.n1, " # Width of the measurements: length of 2nd dimension or number of columns" );
    fprintf ( fw, "%s %i %s\n", "image_size_y: ", params->IM.n2, " # Height of the measurements: length of 1st dimension or number rows" );
    fprintf ( fw, "%s %i %s\n", "image_size_z: ", params->IM.n3, " # Number of image_size_y by image_size_x measurements" );
	fprintf ( fw, "%s %14.8g %s\n", "specimen_tilt_offset_x: ", params->IM.specimen_tilt_offset_x, " # Initial tilt of the object around the second axis [rad]");
	fprintf ( fw, "%s %14.8g %s\n", "specimen_tilt_offset_y: ", params->IM.specimen_tilt_offset_y, " # Initial tilt of the object around the first axis [rad]");
	fprintf ( fw, "%s %14.8g %s\n", "specimen_tilt_offset_z: ", params->IM.specimen_tilt_offset_z, " # Initial tilt of the object around the third axis [rad]");
	fprintf ( fw, "%s %d %s\n", "frozen_phonons: ", params->IM.frPh, " # Number of frozen phonon iterations, set to 0 if none are needed");
	fprintf ( fw, "%s %14.8g %s\n", "pixel_dose: ", params->IM.pD, " # mean number of electrons per pixel, set to zero for noise-free imaging");
	fprintf ( fw, "%s %14.8g %s\n", "subpixel_size_z: ", params->IM.subSlTh, " # The images are calculated with (approx.!) this slice thickness [m]");
    

    fprintf ( fw, "\n# Sample properties\n#------------------\n\n" );
	fprintf ( fw, "%s %14.8g %s\n", "absorptive_potential_factor: ", params->SAMPLE.imPot, " # Imaginary potential factor to approximate absorption: V <- V + iV * absorptive_potential_factor");


    fprintf ( fw, "\n# Specimen tilts, beam tilts and defoci\n#--------------------------------------\n\n" );

	fprintf ( fw, "# Tilts of the specimen [rad]. Quantities in each row:\n" );
	fprintf ( fw, "# specimen_tilt_x,  specimen_tilt_y. Tilts around the second and the first specimen axis resp.\n" );
    for ( int i = 0; i < params->IM.n3; i++ )
    { fprintf ( fw, "%s %14.8g %14.8g\n", "specimen_tilt: ", params->IM.tiltspec[2 * i], params->IM.tiltspec[2 * i + 1] ); }

	fprintf ( fw, "\n# Tilts of the beam [rad].  Quantities in each row:\n" );
	fprintf ( fw, "# beam_tilt_x,  beam_tilt_y. Tilts around the second and the first specimen axis resp.\n" );
    for ( int i = 0; i < params->IM.n3; i++ )
    { fprintf ( fw, "%s %14.8g %14.8g\n", "beam_tilt: ", params->IM.tiltbeam[2 * i], params->IM.tiltbeam[2 * i + 1] ); }

	fprintf ( fw, "\n# Defoci values [m]\n" );
    for ( int i = 0; i < params->IM.n3; i++ )
    { fprintf ( fw, "%s %14.8g\n", "defoci: ", params->IM.defoci[i] ); }


	fprintf ( fw, "\n# CUDA parameters\n#----------------\n\n" );

	fprintf ( fw, "# Grid and block sizes (set automatically)\n" );
	fprintf ( fw, "%s %i %s\n", "gS: ",   params->CU.gS, " # Grid size" );
	fprintf ( fw, "%s %i %s\n", "bS: ",   params->CU.bS , " # Block size");
    
	fprintf ( fw, "\n# The atoms in the sample\n#------------------------\n\n" );
    
	fprintf ( fw, "Number of atoms: %d\n", nAt );

	fprintf ( fw, "\n# List of atoms. Quantities in each row:\n" );
	fprintf ( fw, "# Atomic no.; x-, y- and z-coordinate [m]; Debeye-Waller factor [m^2]; occupancy.\n" );
    for(int j=0; j<nAt; j++)
    {
      fprintf ( fw,"%i %14.8g %14.8g %14.8g %14.8g %14.8g \n",  Z_h[j],  xyzCoord_h[3 * j + 0],  xyzCoord_h[3 * j + 1], xyzCoord_h[3 * j + 2],  DWF_h[j],  occ_h[j] );
	}

    fclose ( fw );

	free( Z_h );
	free( xyzCoord_h );
	free( DWF_h );
	free( occ_h );
}


void printLevelOptimization(params_t* params)
{
//  if (params->IM.m1*params->IM.m2*params->IM.m3>268435456)
//  {
//    printLevel =0;
//  }
}
void defaultParams( params_t* params, int n3 )
{
    params->cst.m0 = 9.1093822e-31f;
    params->cst.c  = 2.9979246e8f;
    params->cst.e  = 1.6021766e-19f;
    params->cst.h  = 6.6260696e-34f;
    params->cst.pi = 3.1415927f;

    params->EM.E0 = 200e3;
    params->EM.gamma = 1.3913902f;
    params->EM.lambda = 2.507934e-012f;
    params->EM.sigma = 7288400.5f;

    params->EM.aberration.C1_0 = -6.1334e-008f;
    params->EM.aberration.A1_0 = 0.f;
    params->EM.aberration.A2_0 = 0.f;
    params->EM.aberration.B2_0 = 0.f;
    params->EM.aberration.C3_0 = 1e-3f;
    params->EM.aberration.A3_0 = 0.f;
    params->EM.aberration.S3_0 = 0.f;
    params->EM.aberration.A4_0 = 0.f;
    params->EM.aberration.B4_0 = 0.f;
    params->EM.aberration.D4_0 = 0.f;
    params->EM.aberration.C5_0 = 0.f;
    params->EM.aberration.A5_0 = 0.f;
    params->EM.aberration.R5_0 = 0.f;
    params->EM.aberration.S5_0 = 0.f;

    params->EM.aberration.C1_1 = 0.f;
    params->EM.aberration.A1_1 = 0.f;
    params->EM.aberration.A2_1 = 0.f;
    params->EM.aberration.B2_1 = 0.f;
    params->EM.aberration.C3_1 = 0.f;
    params->EM.aberration.A3_1 = 0.f;
    params->EM.aberration.S3_1 = 0.f;
    params->EM.aberration.A4_1 = 0.f;
    params->EM.aberration.B4_1 = 0.f;
    params->EM.aberration.D4_1 = 0.f;
    params->EM.aberration.C5_1 = 0.f;
    params->EM.aberration.A5_1 = 0.f;
    params->EM.aberration.R5_1 = 0.f;
    params->EM.aberration.S5_1 = 0.f;

    params->EM.defocspread = 0.f;
    params->EM.illangle = 0.f;
    params->EM.mtfa = 1.f;
    params->EM.mtfb = 0.f;
    params->EM.mtfc = 0.f;
    params->EM.mtfd = 0.f;
    params->EM.ObjAp = 11.1e-3f;

    params->IM.m1 = 4;
    params->IM.m2 = 4;
    params->IM.m3 = 1;
    params->IM.d1 = 0.25e-10f;
    params->IM.d2 = 0.25e-10f;
    params->IM.d3 = 2e-10f;
	params->IM.subSlTh = params->IM.d3;
    params->IM.dn1 = 1;
    params->IM.dn2 = 1;
    params->IM.n1 = 2;
    params->IM.n2 = 2;
    params->IM.n3 = n3;
    params->IM.doBeamTilt = false;
    params->IM.specimen_tilt_offset_x = 0.f;
    params->IM.specimen_tilt_offset_y = 0.f;
    params->IM.specimen_tilt_offset_z = 0.f;
    params->IM.frPh = 0;
	params->IM.pD = 0.f;
	params->IM.mode = 0;
    
    params->SAMPLE.imPot = 0.f;
	params->SAMPLE.nAt = 0;
	sprintf(params->SAMPLE.sample_name, "Empty sample");
    sprintf(params->SAMPLE.material, "Nothing");

    for ( int i = 0; i < n3; i++ )
    {
        params->IM.tiltspec[2 * i] = 0.f;
        params->IM.tiltspec[2 * i + 1] = 0.f;
    }

    for ( int i = 0; i < n3; i++ )
    {
        params->IM.tiltbeam[2 * i] = 0.f;
        params->IM.tiltbeam[2 * i + 1] = 0.f;
    }

    for ( int i = 0; i < n3; i++ )
    { params->IM.defoci[i] = 0.f; }
    
    sprintf(params->USER.user_name, "John Smith");
    sprintf(params->USER.institution, "Europe University");
    sprintf(params->USER.department, "Electron Microscopy Facility");
    sprintf(params->USER.email, "john.smith@uni.eu");
    
	sprintf(params->COMMENT.comments, "This is FDES's default comment");
}

bool getParams(const char* filename, params_t** params, int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d )
{
	bool ret = true;
	params_t* params0;
	allocParams ( &params0, 1000 ); //blocksized memory

	defaultParams( params0, 1000 );
	ret = readConfig (filename, params0,  Z_d, xyzCoord_d, DWF_d, occ_d);

	if(!ret) 
	{
	  
	  return ret;
	  
	};
	const int n3 = params0->IM.n3;

	setGridAndBlockSize ( params0 );
	setCufftPlan ( params0 );
	setCublasHandle ( params0 );

	allocParams ( params, n3 );
	copyParams ( *params, params0 );

	consitentParams ( *params );
	setGridAndBlockSize ( *params );
	setCufftPlan ( *params );
	setCublasHandle ( *params );
	
	const char * fileRead = "dataFDES_used.cnf";
	
 	writeConfig(fileRead, *params, Z_d,  xyzCoord_d, DWF_d, occ_d );

	freeParams ( &params0 );
	return ret;
}

void consitentParams ( params_t* params )
{
    const float E0 = params->EM.E0;
    const float m0 = 9.1093822f;
    const float c  = 2.9979246f;
    const float e  = 1.6021766f;
    const float h  = 6.6260696f;
    const float pi = params->cst.pi;

    params->EM.gamma = 1.f + E0 * e / m0 / c / c * 1e-4f;
    params->EM.lambda = h / sqrt ( 2.f * m0 * e ) * 1e-9f / sqrt ( E0 * ( 1.f + E0 * e / 2.f / m0 / c / c * 1e-4f ) ); // electron wavelength (m), see De Graef p. 92
    params->EM.sigma =  2.f * pi * params->EM.gamma * params->EM.lambda * m0 * e / h / h * 1e18f; // interaction constant (1/(Vm))

    params->IM.m1 = params->IM.n1 + 2 * params->IM.dn1;
    params->IM.m2 = params->IM.n2 + 2 * params->IM.dn2;    
    
    float flag = 0.f;
    for (int j = 0; j < ( params->IM.n3 * 2 ); j++ )
    {    flag += fabsf( params->IM.tiltbeam[j] ); }
    
    if ( flag < ( FLT_MIN * ( (float) params->IM.n3 * 2 ) ) )
      {    params->IM.doBeamTilt = false; }
    else
      {    params->IM.doBeamTilt = true; }


	if ( MATLAB_TILT_COMPATIBILITY == 1 )
	{
		float t_temp = 0.f;
		for ( int k = 0; k < params->IM.n3; k++ )
		{
			t_temp = params->IM.tiltspec[2 * k];
			params->IM.tiltspec[2 * k] = -params->IM.tiltspec[2 * k + 1];
			params->IM.tiltspec[2 * k + 1] = -t_temp;
		}
	}
}


void setCufftPlan ( params_t* params )
{
    cufft_assert ( cufftPlan2d ( & ( params->CU.cufftPlan ), params->IM.m2, params->IM.m1, CUFFT_C2C ) );
}

void setCublasHandle ( params_t* params )
{
    cublas_assert ( cublasCreate ( & ( params->CU.cublasHandle ) ) );
}

void allocParams ( params_t** params, int n3 )
{
	*params = ( params_t* ) malloc ( sizeof ( params_t ) );
	
    if ( !*params )
    {
		fprintf ( stderr, "  Error while allocating params_t" );
        return;
    }
    
    	( **params ).cst.m0  = 9.109389e-31f;  // electron rest mass (kg)
	( **params ).cst.c   = 299792458.0f;   // speed of light (m/s)
	( **params ).cst.e   = 1.602177e-19f;   // elementary charge (C)
	( **params ).cst.h   = 6.626075e-34f;   // Planck's constant (Js)
	( **params ).cst.pi = 3.141592654f;
	
	( **params ).IM.tiltspec = ( float* ) calloc ( 2 * n3, sizeof ( float ) );
	( **params ).IM.tiltbeam = ( float* ) calloc ( 2 * n3, sizeof ( float ) );
	( **params ).IM.defoci   = ( float* ) calloc ( 1 * n3, sizeof ( float ) );
}

void setDeviceParams ( params_t** params_d, params_t* params )
{
    const int n3 = params->IM.n3;

    // allocating and initializing the helper structure on host
    params_t* params_h;
    allocParams ( &params_h, n3 );
    copyParams ( params_h, params );

    // allocating all sub-structures on device
    myCudaMallocSubStruct ( params_h, n3 );

    // Initializing all sub structures
    myCudaMemcpySubStruct ( params_h, params, n3, cudaMemcpyHostToDevice );

    // allocating the structure on device
    cuda_assert ( cudaMalloc ( ( void** ) params_d, sizeof ( params_t ) ) );

    // copying helper struct in it
    cuda_assert ( cudaMemcpy ( *params_d, params_h, sizeof ( params_t ), cudaMemcpyHostToDevice ) );

    // freeing the helper structure
    //myCudaFreeSubStruct(params_h);
    free ( params_h );
}


void getDeviceParams ( params_t** params_h, params_t** params_d, int n3 )
{
    params_t* paramsH_h;
    allocParams ( &paramsH_h, n3 );

    paramsH_h->IM.tiltspec = ( *params_h )->IM.tiltspec;
    paramsH_h->IM.tiltbeam = ( *params_h )->IM.tiltbeam;
    paramsH_h->IM.defoci   = ( *params_h )->IM.defoci;
    cudaMemcpy ( *params_h, *params_d, sizeof ( params_t ), cudaMemcpyDeviceToHost );
    ( *params_h )->IM.tiltspec = paramsH_h->IM.tiltspec;
    ( *params_h )->IM.tiltbeam = paramsH_h->IM.tiltbeam;
    ( *params_h )->IM.defoci   =  paramsH_h->IM.defoci;

    myCudaMallocSubStruct ( paramsH_h, n3 );

    cuda_assert ( cudaMemcpy ( paramsH_h, *params_d, sizeof ( params_t ), cudaMemcpyDeviceToHost ) );

    myCudaMemcpySubStruct ( *params_h, paramsH_h, n3, cudaMemcpyDeviceToHost );

    free ( paramsH_h );
}


void freeParams ( params_t** params )
{
    if ( !*params )
    {
        printf ( "  Error while freeing params_t" );
        return;
    }

    if ( ( **params ).IM.tiltspec )
    {
        free ( ( **params ).IM.tiltspec );
        ( **params ).IM.tiltspec = 0;
    }

    if ( ( **params ).IM.tiltbeam )
    {
        free ( ( **params ).IM.tiltbeam );
        ( **params ).IM.tiltbeam = 0;
    }

    if ( ( **params ).IM.defoci )
    {
        free ( ( **params ).IM.defoci );
        ( **params ).IM.defoci = 0;
    }

    if ( ( **params ).CU.cufftPlan )
    { cufft_assert ( cufftDestroy ( ( **params ).CU.cufftPlan ) ); }

    if ( ( **params ).CU.cublasHandle )
    { cublas_assert ( cublasDestroy ( ( **params ).CU.cublasHandle ) ); }

    free ( *params );
}

void freeDeviceParams ( params_t** params_d, int n3 )
{
    if ( !*params_d )
    {
        fprintf ( stderr, "  Error while cuda-freeing params_t" );
        return;
    }

    params_t* params_h;
    allocParams ( &params_h, n3 );

    myCudaMallocSubStruct ( params_h, n3 );

    cuda_assert ( cudaMemcpy ( params_h, *params_d, sizeof ( params_t ), cudaMemcpyDeviceToHost ) );

    myCudaFreeSubStruct ( params_h );

    cuda_assert ( cudaFree ( *params_d ) );
    free ( params_h );
}

void myCudaMallocSubStruct ( params_t* params, int n3 )
{
    cuda_assert ( cudaMalloc ( ( void** ) & ( params->IM.tiltspec ), 2 * n3 * sizeof ( float ) ) );
    cuda_assert ( cudaMalloc ( ( void** ) & ( params->IM.tiltbeam ), 2 * n3 * sizeof ( float ) ) );
    cuda_assert ( cudaMalloc ( ( void** ) & ( params->IM.defoci ),   1 * n3 * sizeof ( float ) ) );
}

void myCudaFreeSubStruct ( params_t* params )
{
    cuda_assert ( cudaFree ( params->IM.tiltspec ) );
    cuda_assert ( cudaFree ( params->IM.tiltbeam ) );
    cuda_assert ( cudaFree ( params->IM.defoci ) );
}

void myCudaMemcpySubStruct ( params_t* dst, params_t* src, int n3, enum cudaMemcpyKind kind )
{
    cuda_assert ( cudaMemcpy ( dst->IM.tiltspec, src->IM.tiltspec, 2 * n3 * sizeof ( float ), kind ) );
    cuda_assert ( cudaMemcpy ( dst->IM.tiltbeam, src->IM.tiltbeam, 2 * n3 * sizeof ( float ), kind ) );
    cuda_assert ( cudaMemcpy ( dst->IM.defoci,   src->IM.defoci,   1 * n3 * sizeof ( float ), kind ) );
}

void copyParams ( params_t* dst, params_t* src )
{
    dst->cst.m0 = src->cst.m0;
    dst->cst.c  = src->cst.c;
    dst->cst.e  = src->cst.e;
    dst->cst.h  = src->cst.h;
    dst->cst.pi  = src->cst.pi;

    dst->EM.E0 = src->EM.E0;
    dst->EM.gamma = src->EM.gamma;
    dst->EM.lambda = src->EM.lambda;
    dst->EM.sigma = src->EM.sigma;

    dst->EM.aberration.C1_0 = src->EM.aberration.C1_0;
    dst->EM.aberration.A1_0 = src->EM.aberration.A1_0;
    dst->EM.aberration.A2_0 = src->EM.aberration.A2_0;
    dst->EM.aberration.B2_0 = src->EM.aberration.B2_0;
    dst->EM.aberration.C3_0 = src->EM.aberration.C3_0;
    dst->EM.aberration.A3_0 = src->EM.aberration.A3_0;
    dst->EM.aberration.S3_0 = src->EM.aberration.S3_0;
    dst->EM.aberration.A4_0 = src->EM.aberration.A4_0;
    dst->EM.aberration.B4_0 = src->EM.aberration.B4_0;
    dst->EM.aberration.D4_0 = src->EM.aberration.D4_0;
    dst->EM.aberration.C5_0 = src->EM.aberration.C5_0;
    dst->EM.aberration.A5_0 = src->EM.aberration.A5_0;
    dst->EM.aberration.R5_0 = src->EM.aberration.R5_0;
    dst->EM.aberration.S5_0 = src->EM.aberration.S5_0;

    dst->EM.aberration.C1_1 = src->EM.aberration.C1_1;
    dst->EM.aberration.A1_1 = src->EM.aberration.A1_1;
    dst->EM.aberration.A2_1 = src->EM.aberration.A2_1;
    dst->EM.aberration.B2_1 = src->EM.aberration.B2_1;
    dst->EM.aberration.C3_1 = src->EM.aberration.C3_1;
    dst->EM.aberration.A3_1 = src->EM.aberration.A3_1;
    dst->EM.aberration.S3_1 = src->EM.aberration.S3_1;
    dst->EM.aberration.A4_1 = src->EM.aberration.A4_1;
    dst->EM.aberration.B4_1 = src->EM.aberration.B4_1;
    dst->EM.aberration.D4_1 = src->EM.aberration.D4_1;
    dst->EM.aberration.C5_1 = src->EM.aberration.C5_1;
    dst->EM.aberration.A5_1 = src->EM.aberration.A5_1;
    dst->EM.aberration.R5_1 = src->EM.aberration.R5_1;
    dst->EM.aberration.S5_1 = src->EM.aberration.S5_1;

    dst->EM.defocspread = src->EM.defocspread;
    dst->EM.illangle = src->EM.illangle;
    dst->EM.mtfa = src->EM.mtfa;
    dst->EM.mtfb = src->EM.mtfb;
    dst->EM.mtfc = src->EM.mtfc;
    dst->EM.mtfd = src->EM.mtfd;
    dst->EM.ObjAp = src->EM.ObjAp;

    dst->IM.mode = src->IM.mode;
    dst->IM.m1 = src->IM.m1;
    dst->IM.m2 = src->IM.m2;
    dst->IM.m3 = src->IM.m3;
    dst->IM.d1 = src->IM.d1;
    dst->IM.d2 = src->IM.d2;
    dst->IM.d3 = src->IM.d3;
    dst->IM.dn1 = src->IM.dn1;
    dst->IM.dn2 = src->IM.dn2;
    dst->IM.n1 = src->IM.n1;
    dst->IM.n2 = src->IM.n2;
    dst->IM.n3 = src->IM.n3;
    dst->IM.subSlTh = src->IM.subSlTh ;
    dst->IM.pD = src->IM.pD;
    dst->IM.frPh = src->IM.frPh;
    dst->IM.specimen_tilt_offset_x = src->IM.specimen_tilt_offset_x;
    dst->IM.specimen_tilt_offset_y = src->IM.specimen_tilt_offset_y;
    dst->IM.specimen_tilt_offset_z = src->IM.specimen_tilt_offset_z;
    dst->IM.doBeamTilt = src->IM.doBeamTilt;
     
    dst->SAMPLE.nAt=src->SAMPLE.nAt;
    strcpy(dst->SAMPLE.material,src->SAMPLE.material);
    strcpy(dst->SAMPLE.sample_name,src->SAMPLE.sample_name);
    dst->SAMPLE.imPot=src->SAMPLE.imPot;

    
    
    strcpy(dst->USER.user_name,src->USER.user_name);
    strcpy(dst->USER.institution,src->USER.institution);  
    strcpy(dst->USER.department,src->USER.department); 
    strcpy(dst->USER.email,src->USER.email);  
    
    strcpy(dst->COMMENT.comments,src->COMMENT.comments);  
    
    
    
    for ( int i = 0; i < src->IM.n3; i++ )
    {
        dst->IM.tiltspec[2 * i] = src->IM.tiltspec[2 * i];
        dst->IM.tiltspec[2 * i + 1] = src->IM.tiltspec[2 * i + 1];
    }

    for ( int i = 0; i < src->IM.n3; i++ )
    {
        dst->IM.tiltbeam[2 * i] = src->IM.tiltbeam[2 * i];
        dst->IM.tiltbeam[2 * i + 1] = src->IM.tiltbeam[2 * i + 1];
    }

    for ( int i = 0; i < src->IM.n3; i++ )
    { dst->IM.defoci[i] = src->IM.defoci[i]; }

    dst->CU.gS = src->CU.gS;
    dst->CU.gS2D = src->CU.gS2D;
    dst->CU.bS = src->CU.bS;
    dst->CU.cufftPlan = src->CU.cufftPlan;
    dst->CU.cublasHandle = src->CU.cublasHandle;
//     dst->CU.gpu_index= src->CU.gpu_index;
}

void setGridAndBlockSize ( params_t* params )
{
    cudaSetDevice(gpu_index);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpu_index);
  
    const int maxGS = deviceProp.maxGridSize[0]/2 ; // HALF of max gridsize allowed by device, it is taken double elsewhere
    const int maxBS = deviceProp.maxThreadsDim[0]; // Maximum blocksize allowed by device.

    params->CU.bS = maxBS;
    params->CU.gS = ( params->IM.m1 * params->IM.m2 ) / params->CU.bS + 1;

    if ( params->CU.gS > maxGS )
    { params->CU.gS = maxGS; }

    if ( params->CU.bS > maxBS )
    { params->CU.bS = maxBS; }

    if ( params->CU.bS * params->CU.gS < ( ( params->IM.m1 * params->IM.m2 ) ) )
    { fprintf ( stderr, "    WARNING: Dimensions of the object too large for the GPU." ); }

    params->CU.gS2D = params->IM.m1 * params->IM.m2 / params->CU.bS + 1;

	if ( params->CU.bS * params->CU.gS2D < ( ( params->IM.m1 * params->IM.m2 ) ) )
    { fprintf ( stderr, "    WARNING: Dimensions of the object too large for the GPU." ); }
    
}

void transposeMeasurements ( float* I, params_t* params )
{
    // Naive transpose, doesn't matter, happens only in the beginning and at the end
    const int n1 = params->IM.n1;
    const int n2 = params->IM.n2;
    const int n3 = params->IM.n3;

    float* J;
    J = ( float* ) malloc ( n1 * n2 * sizeof ( float ) );

    for ( int i3 = 0; i3 < n3; i3++ )
    {
        for ( int i2 = 0; i2 < n2; i2++ )
        {
            for ( int i1 = 0; i1 < n1; i1++ )
            { J[n2 * i1 + i2] = I[i3 * n1 * n2 + i2 * n1 + i1]; }
        }

        for ( int k = 0; k < n1 * n2; k++ )
        { I[i3 * n1 * n2 + k] = J[k]; }
    }

    free ( J );

    params->IM.n1 = n2;
    params->IM.n2 = n1;
    int temp = params->IM.m1;
    params->IM.m1 = params->IM.m2;
    params->IM.m2 = temp;
    temp = params->IM.dn1;
    params->IM.dn1 = params->IM.dn2;
    params->IM.dn2 = temp;
    float tempf = params->IM.d1;
    params->IM.d1 = params->IM.d2;
    params->IM.d2 = tempf;

    for ( int k = 0; k < n3; k++ )
    {
        tempf = params->IM.tiltspec[2 * k];
        params->IM.tiltspec[2 * k] = params->IM.tiltspec[2 * k + 1];
        params->IM.tiltspec[2 * k + 1] = tempf;
        tempf = params->IM.tiltbeam[2 * k];
        params->IM.tiltbeam[2 * k] = params->IM.tiltbeam[2 * k + 1];
        params->IM.tiltbeam[2 * k + 1] = tempf;
    }
}

int numberOfAtoms( FILE* fr )
{
	fseek ( fr, 0, SEEK_SET );
	int nAt = 0;
	const int size = 200;
	char* line;
	line = ( char* ) malloc ( size * sizeof ( char ) );
	char* fieldName;
	fieldName = ( char* ) malloc ( size * sizeof ( char ) );

	while ( !feof( fr ) )
	{
		if(fgets ( line, size, fr )!=NULL)
		sscanf ( line, "%s", fieldName );

		if ( !strncmp ( fieldName, "atom:", 5 ) )
		{    
			nAt += 1; 
			resetLine( line );
		}

	}

	return( nAt );
}

void resetLine( char* line )
{
	line[0] = *( "#" ); // Reset "loremipsum" to "#oremipsum", indicating a comment.
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

	while ( !feof( fr ) )
	{
		if(fgets ( line, size, fr )!=NULL)
		sscanf ( line, "%s", fieldName );

		if ( !strncmp ( fieldName, "atom:", 5 ) )
		{    
			sscanf ( line, "%*s %i %g %g %g %g %g", & Z_h[j], & xyzCoord_h[3 * j + 0], & xyzCoord_h[3 * j + 1], & xyzCoord_h[3 * j + 2], & DWF_h[j], & occ_h[j] );
			j += 1;
		}

		resetLine ( line );

	}

	free ( line );
	free ( fieldName );
}

