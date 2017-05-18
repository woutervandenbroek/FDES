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


void readConfig ( char* file, params_t* params )
{
	FILE* fr;
	const int size = 256;
	char* line;
	line = ( char* ) malloc ( size * sizeof ( char ) );
	char* fieldname;
	fieldname = ( char* ) malloc ( size * sizeof ( char ) );
	int ts_i = 0;
	int tb_i = 0;
	int df_i = 0;
	int nAt = 0;

	fr = fopen ( file, "rt" );

	do
	{
		fgets ( line, size, fr );
		sscanf ( line, "%s", fieldname );

		if ( !strncmp ( fieldname, "voltage:", 8 ) ){
			sscanf ( line, "%*s %g", & ( *params ).EM.E0 ); 
		}
		if ( !strncmp ( fieldname, "C1:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.C1_0, & ( *params ).EM.aberration.C1_1 ); 
		}
		if ( !strncmp ( fieldname, "A1:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.A1_0, & ( *params ).EM.aberration.A1_1 ); 
		}
		if ( !strncmp ( fieldname, "A2:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.A2_0, & ( *params ).EM.aberration.A2_1 ); 
		}
		if ( !strncmp ( fieldname, "B2:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.B2_0, & ( *params ).EM.aberration.B2_1 ); 
		}
		if ( !strncmp ( fieldname, "C3:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.C3_0, & ( *params ).EM.aberration.C3_1 ); 
		}
		if ( !strncmp ( fieldname, "A3:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.A3_0, & ( *params ).EM.aberration.A3_1 );	
		}
		if ( !strncmp ( fieldname, "S3:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.S3_0, & ( *params ).EM.aberration.S3_1 ); 
		}
		if ( !strncmp ( fieldname, "A4:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.A4_0, & ( *params ).EM.aberration.A4_1 ); 
		}
		if ( !strncmp ( fieldname, "B4:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.B4_0, & ( *params ).EM.aberration.B4_1 ); 
		}
		if ( !strncmp ( fieldname, "D4:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.D4_0, & ( *params ).EM.aberration.D4_1 ); 
		}
		if ( !strncmp ( fieldname, "C5:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.C5_0, & ( *params ).EM.aberration.C5_1 ); 
		}
		if ( !strncmp ( fieldname, "A5:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.A5_0, & ( *params ).EM.aberration.A5_1 ); 
		}
		if ( !strncmp ( fieldname, "R5:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.R5_0, & ( *params ).EM.aberration.R5_1 ); 
		}
		if ( !strncmp ( fieldname, "S5:", 3 ) )	{
			sscanf ( line, "%*s %g %g", & ( *params ).EM.aberration.S5_0, & ( *params ).EM.aberration.S5_1 ); 
		}
		if ( !strncmp ( fieldname, "focus_spread:", 13 ) ) {
			sscanf ( line, "%*s %g", & ( *params ).EM.defocspread ); 
		}
		if ( !strncmp ( fieldname, "illumination_angle:", 19 ) ) {
			sscanf ( line, "%*s %g", & ( *params ).EM.illangle ); 
		}
		if ( !strncmp ( fieldname, "mtf_a:", 6 ) ) {
			sscanf ( line, "%*s %g", & ( *params ).EM.mtfa ); 
		}
		if ( !strncmp ( fieldname, "mtf_b:", 6 ) ) {
			sscanf ( line, "%*s %g", & ( *params ).EM.mtfb ); 
		}
		if ( !strncmp ( fieldname, "mtf_c:", 6 ) ) {
			sscanf ( line, "%*s %g", & ( *params ).EM.mtfc ); 
		}
		if ( !strncmp ( fieldname, "mtf_d:", 6 ) ) {
			sscanf ( line, "%*s %g", & ( *params ).EM.mtfd ); 
		}
		if ( !strncmp ( fieldname, "objective_aperture:", 19 ) ) {
			sscanf ( line, "%*s %g", & ( *params ).EM.ObjAp ); 
		}
		if ( !strncmp ( fieldname, "condensor_angle:", 16 ) ) {
			sscanf ( line, "%*s %g", & ( *params ).EM.condensorAngle ); 
		}
		if ( !strncmp ( fieldname, "sample_size_x:", 14 ) ) {
			sscanf ( line, "%*s %i", & ( *params ).IM.m1 ); 
		}
		if ( !strncmp ( fieldname, "sample_size_y:", 14 ) )	{
			sscanf ( line, "%*s %i", & ( *params ).IM.m2 );	
		}
		if ( !strncmp ( fieldname, "sample_size_z:", 14 ) )	{
			sscanf ( line, "%*s %i", & ( *params ).IM.m3 ); 
		}
		if ( !strncmp ( fieldname, "pixel_size_x:", 13 ) )	{
			sscanf ( line, "%*s %g", & ( *params ).IM.d1 ); 
		}
		if ( !strncmp ( fieldname, "pixel_size_y:", 13 ) )	{
			sscanf ( line, "%*s %g", & ( *params ).IM.d2 ); 
		}
		if ( !strncmp ( fieldname, "pixel_size_z:", 13 ) )	{
			sscanf ( line, "%*s %g", & ( *params ).IM.d3 ); 
		}
		if ( !strncmp ( fieldname, "border_size_x:", 14 ) ) {
			sscanf ( line, "%*s %i", & ( *params ).IM.dn1 ); 
		}
		if ( !strncmp ( fieldname, "border_size_y:", 14 ) ) {
			sscanf ( line, "%*s %i", & ( *params ).IM.dn2 ); 
		}
		if ( !strncmp ( fieldname, "image_size_x:", 13 ) ) {
			sscanf ( line, "%*s %i", & ( *params ).IM.n1 ); 
		}
		if ( !strncmp ( fieldname, "image_size_y:", 13 ) )	{
			sscanf ( line, "%*s %i", & ( *params ).IM.n2 ); 
		}
		if ( !strncmp ( fieldname, "image_size_z:", 13 ) )	{
			sscanf ( line, "%*s %i", & ( *params ).IM.n3 ); 
		}
		if ( !strncmp ( fieldname, "specimen_tilt:", 14 ) ) {
			sscanf ( line, "%*s %g %g", & ( *params ).IM.tiltspec[2 * ts_i], & ( *params ).IM.tiltspec[2 * ts_i + 1] );
			ts_i += 1;
		}
		if ( !strncmp ( fieldname, "beam_tilt:", 10 ) ) {
			sscanf ( line, "%*s %g %g", & ( *params ).IM.tiltbeam[2 * tb_i], & ( *params ).IM.tiltbeam[2 * tb_i + 1] );
			tb_i += 1;
		}
		if ( !strncmp ( fieldname, "defoci:", 7 ) ) {
			sscanf ( line, "%*s %g", & ( *params ).IM.defoci[df_i] );
			df_i += 1;
		}
		if ( !strncmp ( fieldname, "absorptive_potential_factor:", 28 )) {
			sscanf ( line, "%*s %g", &(*params).SAMPLE.imPot);
		}
		if ( !strncmp ( fieldname, "pixel_dose:", 11 ) ) {
			sscanf ( line, "%*s %g", &(*params).IM.pD);
		}
		if ( !strncmp ( fieldname, "frozen_phonons:", 15 ) ) {
			sscanf ( line, "%*s %d", &(*params).IM.frPh);
		}			
		if ( !strncmp ( fieldname, "subpixel_size_z:", 16 ) ) {
			sscanf ( line, "%*s %g", &(*params).IM.subSlTh);
		}
		if ( !strncmp ( fieldname, "specimen_tilt_offset_x:", 23 ) ) {
			sscanf ( line, "%*s %g", &(*params).IM.specimen_tilt_offset_x);
		}		
		if ( !strncmp ( fieldname, "specimen_tilt_offset_y:", 23 ) ) {
			sscanf ( line, "%*s %g", &(*params).IM.specimen_tilt_offset_y);
		}				
		if ( !strncmp ( fieldname, "specimen_tilt_offset_z:", 23 ) ) {
			sscanf ( line, "%*s %g", &(*params).IM.specimen_tilt_offset_z);
		}		
		if ( !strncmp ( fieldname, "mode:", 5 ) ) {
			sscanf ( line, "%*s %d", &(*params).IM.mode);
		}
		if ( !strncmp ( fieldname, "gpu_index:", 10 ) ) {
			sscanf ( line, "%*s %d", &gpu_index );
		}
		if ( !strncmp ( fieldname, "atom:", 5 ) ) {    
			nAt += 1;
		}
		if ( !strncmp ( fieldname, "inner_AD_angle:", 15 ) ) {
			sscanf ( line, "%*s %g", &(*params).SCN.thIn );
		}
		if ( !strncmp ( fieldname, "outer_AD_angle:", 15 ) ) {
			sscanf ( line, "%*s %g", &(*params).SCN.thOut );
		}
		if ( !strncmp ( fieldname, "scan_width_x:", 13 ) ) {
			sscanf ( line, "%*s %d", &(*params).SCN.o1 );
		}
		if ( !strncmp ( fieldname, "scan_width_y:", 13 ) ) {
			sscanf ( line, "%*s %d", &(*params).SCN.o2 );
		}
		if ( !strncmp ( fieldname, "scan_step_x:", 12 ) ) {
			sscanf ( line, "%*s %g", &(*params).SCN.dSx );
		}
		if ( !strncmp ( fieldname, "scan_step_y:", 12 ) ) {
			sscanf ( line, "%*s %g", &(*params).SCN.dSy );
		}
		if ( !strncmp ( fieldname, "scan_center_x:", 14 ) ) {
			sscanf ( line, "%*s %g", &(*params).SCN.c1 );
		}
		if ( !strncmp ( fieldname, "scan_center_y:", 14 ) ) {
			sscanf ( line, "%*s %g", &(*params).SCN.c2 );
		}
		if ( !strncmp ( fieldname, "scan_contrast_factor:", 21 ) ) {
			sscanf ( line, "%*s %g", &(*params).SCN.cF );
		}
		line[0] = *( "#" );
	} while ( !feof( fr ) );

	params->SAMPLE.nAt = nAt;

	fclose ( fr );
	free ( line );
	free ( fieldname );
}


int readCudaDeviceRank( )
{
	// Reads the rank of the Cuda device. Defaults to 0.
	FILE* fr;
	const int size = 256;
	char* line;
	line = ( char* ) malloc ( size * sizeof ( char ) );
	char* fieldname;
	fieldname = ( char* ) malloc ( size * sizeof ( char ) );

	fr = fopen ( "Params.cnf", "rt" );

	gpu_index = 0;
	do {
		fgets ( line, size, fr );
		sscanf ( line, "%s", fieldname );

		if ( !strncmp ( fieldname, "gpu_index:", 10 ) ){
			sscanf ( line, "%*s %d", &gpu_index ); 
		}
	} while ( !feof( fr ) );

	fclose ( fr );
	free ( line );
	free ( fieldname );

	return( gpu_index );
}


void writeConfig ( const char* file, params_t* params, int* Z_d, float* xyzCoord_d, float* DWF_d, float* occ_d )
{
	int nAt = params->SAMPLE.nAt;
	
	int *Z_h;
	float *xyzCoord_h, *DWF_h, *occ_h;
	Z_h = ( int* ) malloc ( nAt * sizeof ( int ) );
	xyzCoord_h = ( float* ) malloc ( nAt * 3 * sizeof ( float ) );
	DWF_h = ( float* ) malloc ( nAt * sizeof ( float ) );
	occ_h = ( float* ) malloc ( nAt * sizeof ( float ) );

	cuda_assert ( cudaMemcpy ( Z_h, Z_d, nAt * sizeof ( int ), cudaMemcpyDeviceToHost) );
	cuda_assert ( cudaMemcpy ( xyzCoord_h, xyzCoord_d, nAt * 3 * sizeof ( float ), cudaMemcpyDeviceToHost ) );
	cuda_assert ( cudaMemcpy ( DWF_h, DWF_d, nAt * sizeof ( float ), cudaMemcpyDeviceToHost ) );
	cuda_assert ( cudaMemcpy ( occ_h, occ_d, nAt * sizeof ( float ), cudaMemcpyDeviceToHost ) );

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
	fprintf ( fw, "%s %14.8g %s\n", "condensor_angle: ", params->EM.condensorAngle, " # Half-angle of convergence of the illumination [rad]" );

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

	fprintf ( fw, "%s %i %s\n", "mode: ", params->IM.mode, " # 0, 1, 2 or 3 for imaging, diffraction, CBED or STEM, resp." );
	fprintf ( fw, "%s %i %s\n", "gpu_index: ", gpu_index, " # Rank of the device the code runs on, it's often 0 or 1" );
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

	fprintf ( fw, "\n# Scan settings\n#------------------\n\n" );
	fprintf ( fw, "%s %14.8g %s\n", "inner_AD_angle: ", params->SCN.thIn, " # Inner radius of the annular detector [rad]");
	fprintf ( fw, "%s %14.8g %s\n", "outer_AD_angle: ", params->SCN.thOut, " # Outer radius of the annular detector [rad]");
	fprintf ( fw, "%s %i %s\n", "scan_width_x: ", params->SCN.o1, " # 2nd dimension of the scan [pix]");
	fprintf ( fw, "%s %i %s\n", "scan_width_y: ", params->SCN.o2, " # 1st dimension of the scan [pix]");
	fprintf ( fw, "%s %14.8g %s\n", "scan_step_x: ", params->SCN.dSx, " # Sampling length in 2nd dimension [m]");
	fprintf ( fw, "%s %14.8g %s\n", "scan_step_y: ", params->SCN.dSy, " # Sampling length in 1st dimension [m]");
	fprintf ( fw, "%s %14.8g %s\n", "scan_center_x: ", params->SCN.c1, " # Center of the scan in 2nd dimension [m]");
	fprintf ( fw, "%s %14.8g %s\n", "scan_center_y: ", params->SCN.c2, " # Center of the scan in 1st dimension [m]");
	fprintf ( fw, "%s %14.8g %s\n", "scan_contrast_factor: ", params->SCN.cF, " # Contrast factor [dimensionless], see Ultramicroscopy 159 (2015) 46--58");

	fprintf ( fw, "\n# Sample properties\n#------------------\n\n" );
	fprintf ( fw, "%s %14.8g %s\n", "absorptive_potential_factor: ", params->SAMPLE.imPot, " # Imaginary potential factor to approximate absorption: V <- V + iV * absorptive_potential_factor");
	fprintf ( fw, "%s %d %s\n", "number_atoms: ", params->SAMPLE.nAt, " # Total number of atoms in the sample");

	fprintf ( fw, "\n# Specimen tilts, beam tilts and defoci\n#--------------------------------------\n\n" );

	fprintf ( fw, "# Tilts of the specimen [rad]. Quantities in each row:\n" );
	fprintf ( fw, "# specimen_tilt_x,  specimen_tilt_y. Tilts around the second and the first specimen axis resp.\n" );
	for ( int i = 0; i < params->IM.n3; i++ ) { 
		fprintf ( fw, "%s %14.8g %14.8g\n", "specimen_tilt: ", params->IM.tiltspec[2 * i], params->IM.tiltspec[2 * i + 1] ); 
	}
	fprintf ( fw, "\n# Tilts of the beam [rad].  Quantities in each row:\n" );
	fprintf ( fw, "# beam_tilt_x,  beam_tilt_y. Tilts around the first and the second specimen axis resp.\n" );
	for ( int i = 0; i < params->IM.n3; i++ ) { 
		fprintf ( fw, "%s %14.8g %14.8g\n", "beam_tilt: ", params->IM.tiltbeam[2 * i], params->IM.tiltbeam[2 * i + 1] ); 
	}
	fprintf ( fw, "\n# Defoci values [m]\n" );
	for ( int i = 0; i < params->IM.n3; i++ ) { 
		fprintf ( fw, "%s %14.8g\n", "defoci: ", params->IM.defoci[i] ); 
	}

	fprintf ( fw, "\n# CUDA parameters\n#----------------\n\n" );
	fprintf ( fw, "# Grid and block sizes (set automatically)\n" );
	fprintf ( fw, "%s %i %s\n", "gS: ",   params->CU.gS, " # Grid size" );
	fprintf ( fw, "%s %i %s\n", "bS: ",   params->CU.bS , " # Block size");

	fprintf ( fw, "\n# List of atoms. Quantities in each row:\n" );
	fprintf ( fw, "# Atomic no.; x-, y- and z-coordinate [m]; Debeye-Waller factor [m^2]; occupancy.\n" );
	for(int j=0; j<nAt; j++) {
		fprintf ( fw,"%s %i %14.8g %14.8g %14.8g %14.8g %14.8g \n",  "atom: ", Z_h[j],  xyzCoord_h[3 * j + 0],  xyzCoord_h[3 * j + 1], xyzCoord_h[3 * j + 2],  DWF_h[j],  occ_h[j] );
	}

	free( Z_h );
	free( xyzCoord_h );
	free( DWF_h );
	free( occ_h );

	fclose ( fw );
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
	params->EM.condensorAngle = 0.f;
	params->EM.mtfa = 1.f;
	params->EM.mtfb = 0.f;
	params->EM.mtfc = 0.f;
	params->EM.mtfd = 0.f;
	params->EM.ObjAp = 1.57f;

	params->IM.mode = 0;
	params->IM.m1 = 4;
	params->IM.m2 = 4;
	params->IM.m3 = 1;
	params->IM.d1 = 0.25e-10f;
	params->IM.d2 = 0.25e-10f;
	params->IM.d3 = 2e-10f;
	params->IM.dn1 = 1;
	params->IM.dn2 = 1;
	params->IM.n1 = 2;
	params->IM.n2 = 2;
	params->IM.n3 = n3;
	for ( int i = 0; i < n3; i++ ) {
		params->IM.tiltspec[2 * i] = 0.f;
		params->IM.tiltspec[2 * i + 1] = 0.f;
	}
	for ( int i = 0; i < n3; i++ ) {
		params->IM.tiltbeam[2 * i] = 0.f;
		params->IM.tiltbeam[2 * i + 1] = 0.f;
	}
	for ( int i = 0; i < n3; i++ ) {
		params->IM.defoci[i] = 0.f; 
	}
	gpu_index = 0;

	params->IM.specimen_tilt_offset_x = 0.f;
	params->IM.specimen_tilt_offset_y = 0.f;
	params->IM.specimen_tilt_offset_z = 0.f;
	params->IM.frPh = 0;
	params->IM.pD = 0.f;
	params->IM.subSlTh = 2e-10f;
	params->IM.doBeamTilt = false;

	params->SAMPLE.imPot = 0.f;
	params->SAMPLE.nAt = 0;

	params->SCN.thIn = 0.f;
	params->SCN.thOut = 1.57f;
	params->SCN.o1 = 1;
	params->SCN.o2 = 1;
	params->SCN.dSx = 0.25e-10f;
	params->SCN.dSy = 0.25e-10f;
	params->SCN.c1 = 0.f; 
	params->SCN.c2 = 0.f; 
	params->SCN.cF = 1.f;
 }


void getParams( params_t** params )
{
	params_t* params0;
	allocParams ( &params0, 1000 );

	defaultParams( params0, 1000 );

	readConfig ( "Params.cnf", params0 );
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

	freeParams ( &params0 );
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
	params->EM.condensorAngle = fabsf( params->EM.condensorAngle );
	params->EM.ObjAp = fabsf( params->EM.ObjAp );

	params->IM.m1 = params->IM.n1 + 2 * params->IM.dn1;
	params->IM.m2 = params->IM.n2 + 2 * params->IM.dn2;
	
	if ( MATLAB_TILT_COMPATIBILITY == 1 ) {
		float t_temp = 0.f;
		for ( int k = 0; k < params->IM.n3; k++ ) {
			t_temp = params->IM.tiltspec[2 * k];
			params->IM.tiltspec[2 * k] = -params->IM.tiltspec[2 * k + 1];
			params->IM.tiltspec[2 * k + 1] = -t_temp;
		}
	}
	
	if( params->EM.aberration.A1_0 < 0.f ) {
		params->EM.aberration.A1_0 = fabsf( params->EM.aberration.A1_0 );
		params->EM.aberration.A1_1 += params->cst.pi;
	}
	if( params->EM.aberration.A2_0 < 0.f ) {
		params->EM.aberration.A2_0 = fabsf( params->EM.aberration.A2_0 );
		params->EM.aberration.A2_1 += params->cst.pi;
	}
	if( params->EM.aberration.B2_0 < 0.f ) {
		params->EM.aberration.B2_0 = fabsf( params->EM.aberration.B2_0 );
		params->EM.aberration.B2_1 += params->cst.pi;
	}
	float tmp = 0.f;
	for ( int i = 0; i < ( 2*params->IM.n3 ); i++ ) {
		tmp += fabsf( params->IM.tiltbeam[i] );
	}
	params->IM.doBeamTilt = false;
	if ( tmp >  1e-4f ) {
		params->IM.doBeamTilt = true;
	}
	if ( params->IM.doBeamTilt ){ // Round the beam tilts to multiples of the sampling in Fourier space to enforce periodic boundary conditions
		float dTh1 = asinf( params->EM.lambda / ( params->IM.d1 * ( (float) params->IM.m1 ) ) );
		float dTh2 = asinf( params->EM.lambda / ( params->IM.d2 * ( (float) params->IM.m2 ) ) );
		for ( int i = 0; i < params->IM.n3; i++ ) {
			params->IM.tiltbeam[2*i]   = roundf( params->IM.tiltbeam[2*i]   / dTh1 ) * dTh1;
			params->IM.tiltbeam[2*i+1] = roundf( params->IM.tiltbeam[2*i+1] / dTh2 ) * dTh2;
		}
	}
	if ( params->IM.mode != 3 ) {
		params->SCN.o1 = 1;
		params->SCN.o2 = 1;
	}
}


void setCufftPlan ( params_t* params )
{
    cufft_assert ( cufftPlan2d ( & ( params->CU.cufftPlan ), params->IM.m1, params->IM.m2, CUFFT_C2C ) );
}


void setCublasHandle ( params_t* params )
{
    cublas_assert ( cublasCreate ( & ( params->CU.cublasHandle ) ) );
}

void allocParams ( params_t** params, int n3 )
{
	*params = ( params_t* ) malloc ( sizeof ( params_t ) );

	if ( !*params ) {
		fprintf ( stderr, "  Error while allocating params_t" );
		return;
	}
	
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

	//myCudaFreeSubStruct(paramsH_h);

	free ( paramsH_h );
}


void freeParams ( params_t** params )
{
	if ( !*params )	{
		printf ( "  Error while freeing params_t" );
		return;
	}
	if ( ( **params ).IM.tiltspec ) {
		free ( ( **params ).IM.tiltspec );
		( **params ).IM.tiltspec = 0;
	}
	if ( ( **params ).IM.tiltbeam ) {
		free ( ( **params ).IM.tiltbeam );
		( **params ).IM.tiltbeam = 0;
	}
	if ( ( **params ).IM.defoci ) {
		free ( ( **params ).IM.defoci );
		( **params ).IM.defoci = 0;
	}
	if ( ( **params ).CU.cufftPlan ) {
		cufft_assert ( cufftDestroy ( ( **params ).CU.cufftPlan ) ); 
	}
	if ( ( **params ).CU.cublasHandle ) {
		cublas_assert ( cublasDestroy ( ( **params ).CU.cublasHandle ) ); 
	}

	free ( *params );
}

void freeDeviceParams ( params_t** params_d, int n3 )
{
	if ( !*params_d ) {
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
	dst->EM.condensorAngle = src->EM.condensorAngle;
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
	for ( int i = 0; i < ( 2*src->IM.n3 ); i++ ) {
		dst->IM.tiltspec[i] = src->IM.tiltspec[i];
	}
	for ( int i = 0; i < ( 2*src->IM.n3 ); i++ ) {
		dst->IM.tiltbeam[i] = src->IM.tiltbeam[i];
	}
	for ( int i = 0; i < src->IM.n3; i++ ) {
		dst->IM.defoci[i] = src->IM.defoci[i]; 
	}
	dst->IM.frPh = src->IM.frPh;
	dst->IM.pD = src->IM.pD;
	dst->IM.subSlTh = src->IM.subSlTh;
	dst->IM.specimen_tilt_offset_x = src->IM.specimen_tilt_offset_x;
	dst->IM.specimen_tilt_offset_y = src->IM.specimen_tilt_offset_y;
	dst->IM.specimen_tilt_offset_z = src->IM.specimen_tilt_offset_z;
	dst->IM.doBeamTilt = src->IM.doBeamTilt;

	dst->CU.gS = src->CU.gS;
	dst->CU.gS2D = src->CU.gS2D;
	dst->CU.bS = src->CU.bS;
	dst->CU.cufftPlan = src->CU.cufftPlan;
	dst->CU.cublasHandle = src->CU.cublasHandle;

	dst->SAMPLE.imPot = src->SAMPLE.imPot;
	dst->SAMPLE.nAt = src->SAMPLE.nAt;

	dst->SCN.thIn = src->SCN.thIn;
	dst->SCN.thOut = src->SCN.thOut;
	dst->SCN.o1 = src->SCN.o1;
	dst->SCN.o2 = src->SCN.o2;
	dst->SCN.dSx = src->SCN.dSx;
	dst->SCN.dSy = src->SCN.dSy;
	dst->SCN.c1 = src->SCN.c1; 
	dst->SCN.c2 = src->SCN.c2; 
	dst->SCN.cF = src->SCN.cF;
}

void setGridAndBlockSize ( params_t* params )
{
	const int maxGS = 1073741823; // HALF of max gridsize allowed by device, it is taken double elsewhere
	const int maxBS = 1024; // Maximum blocksize allowed by device.

	params->CU.bS = 1024;
	params->CU.gS = ( params->IM.m1 * params->IM.m2 * ( params->IM.m3 + 4 ) ) / params->CU.bS + 1;

	if ( params->CU.gS > maxGS ) {
		params->CU.gS = maxGS; 
	}

	if ( params->CU.bS > maxBS ) {
		params->CU.bS = maxBS; 
	}

	// if ( params->CU.bS * params->CU.gS < ( ( params->IM.m1 * params->IM.m2 * params->IM.m3 ) ) ) {
	if ( params->CU.bS * params->CU.gS < ( ( params->IM.m1 * params->IM.m2 ) ) ) {
		fprintf ( stderr, "    WARNING: Dimensions of the object too large for the GPU." ); }

	params->CU.gS2D = params->IM.m1 * params->IM.m2 / params->CU.bS + 1;

	if ( params->CU.bS * params->CU.gS2D < ( ( params->IM.m1 * params->IM.m2 ) ) ) {
		fprintf ( stderr, "    WARNING: Dimensions of the object too large for the GPU." ); 
	}
}

void transposeMeasurements ( float* I, params_t* params )
{
	// Naive transpose, doesn't matter, it'll be used only once
	const int n1 = params->IM.n1;
	const int n2 = params->IM.n2;
	const int n3 = params->IM.n3;

	float* J;
	J = ( float* ) malloc ( n1 * n2 * sizeof ( float ) );

	for ( int i3 = 0; i3 < n3; i3++ ) {
		for ( int i2 = 0; i2 < n2; i2++ ) {
			for ( int i1 = 0; i1 < n1; i1++ ) {
				J[n2 * i1 + i2] = I[i3 * n1 * n2 + i2 * n1 + i1]; 
			}
		}
		for ( int k = 0; k < n1 * n2; k++ ) {
			I[i3 * n1 * n2 + k] = J[k]; 
		}
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

	for ( int k = 0; k < n3; k++ ) {
		tempf = params->IM.tiltspec[2 * k];
		params->IM.tiltspec[2 * k] = params->IM.tiltspec[2 * k + 1];
		params->IM.tiltspec[2 * k + 1] = tempf;
		tempf = params->IM.tiltbeam[2 * k];
		params->IM.tiltbeam[2 * k] = params->IM.tiltbeam[2 * k + 1];
		params->IM.tiltbeam[2 * k + 1] = tempf;
	}
}
