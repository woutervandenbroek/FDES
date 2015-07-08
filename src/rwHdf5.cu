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

#include "rwHdf5.h"

void writeHdf5 (const char* filename, float* image,float * potential, float * exitwave, params_t* params,int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d)
{
	int nAt = params->SAMPLE.nAt;
    	int *Z_h;
	float *xyzCoord_h, *DWF_h, *occ_h;
	float *xCoord_h, *yCoord_h, *zCoord_h;
	Z_h = ( int* ) malloc ( nAt * sizeof ( int ) );
	xyzCoord_h = ( float* ) malloc ( nAt * 3 * sizeof ( float ) );
	xCoord_h = ( float* ) malloc ( nAt *  sizeof ( float ) );
	yCoord_h = ( float* ) malloc ( nAt *  sizeof ( float ) );
	zCoord_h = ( float* ) malloc ( nAt *  sizeof ( float ) );
	DWF_h = ( float* ) malloc ( nAt * sizeof ( float ) );
	occ_h = ( float* ) malloc ( nAt * sizeof ( float ) );

	cuda_assert ( cudaMemcpy ( Z_h, *Z_d, nAt * sizeof ( int ), cudaMemcpyDeviceToHost) );
	cuda_assert ( cudaMemcpy ( xyzCoord_h, *xyzCoord_d, nAt * 3 * sizeof ( int ), cudaMemcpyDeviceToHost ) );
	cuda_assert ( cudaMemcpy ( DWF_h, *DWF_d, nAt * sizeof ( float ), cudaMemcpyDeviceToHost ) );
	cuda_assert ( cudaMemcpy ( occ_h, *occ_d, nAt * sizeof ( float ), cudaMemcpyDeviceToHost ) );

        for (int i =0;i < nAt; i++)
	{
	  xCoord_h[i]= xyzCoord_h[i*3+0];
	  yCoord_h[i]= xyzCoord_h[i*3+1];
	  zCoord_h[i]= xyzCoord_h[i*3+2];
	  
	}
  
	hid_t   file;         /* file and dataset handles */
	file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
 
/////////////////////////////// attribute "version"
	float version_value= version;
	writeAttributeSingle(file,"version",version_value);  
///////////////////////////// group  "data"
	hid_t grp_data;
	grp_data = H5Gcreate2(file, "/data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
	if (printLevel>0 && params->IM.m1*params->IM.m2*params->IM.m3<268435456)
	{
///////////////////////////////// subgroup of "data" potential_slices          

	hid_t grp_potential_slice;
	grp_potential_slice = H5Gcreate2(grp_data, "potential_slices", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
///////////////////////////// group "potential_slices" attribute "emd_group_type"    
	
	unsigned char emd_group_type_value = 1;
	writeAttributeSingle(grp_potential_slice,"emd_group_type",emd_group_type_value);  
	
///////////////////////////// group "potential_slices" dataset "data"
	
	float * p_xyz;
	p_xyz = ( float* ) malloc ( params->IM.m1* params->IM.m2*  params->IM.m3 * 2*  sizeof ( float ) );
   
	for (int i= 0; i <params->IM.m3; i++)
	    for (int j= 0; j <params->IM.m2; j++)
	      for (int k= 0; k <params->IM.m1; k++)
		{
		  p_xyz[2*(k*params->IM.m3 *params->IM.m2 + j *params->IM.m3 + i) + 0 ] = potential[ 2 *(i*params->IM.m1 *params->IM.m2 + j *params->IM.m1 + k) + 0  ];
		  p_xyz[2*(k*params->IM.m3 *params->IM.m2 + j *params->IM.m3 + i) + 1 ] = potential[ 2 *(i*params->IM.m1 *params->IM.m2 + j *params->IM.m1 + k) + 1  ];
		}

	hid_t  dsp_potential_data;
	dsp_potential_data = H5Screate(H5S_SIMPLE);
	int data_potential_RANK = 4;
	hsize_t data_potential_dim[] = {(long unsigned int)params->IM.m1, (long unsigned int)params->IM.m2, (long unsigned int) params->IM.m3, 2};
	H5Sset_extent_simple(dsp_potential_data, data_potential_RANK, data_potential_dim, NULL);
	hid_t dst_potential_data;
	dst_potential_data = H5Dcreate2(grp_potential_slice, "data", H5T_NATIVE_FLOAT, dsp_potential_data, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(dst_potential_data, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, p_xyz);
	H5Sclose(dsp_potential_data); 
	H5Dclose(dst_potential_data);
    
/////////////////////////// group "potential_slices" dataset "dim1" 
   
	float * potential_x_coordinates;
	potential_x_coordinates = ( float* ) malloc ( params->IM.m1 *  sizeof ( float ) );
 	for (int i=0; i<params->IM.m1; i++)
	{
	  potential_x_coordinates[i] = i -(params->IM.m1-1)/2.0;
	}
   
	hid_t  dsp_potential_dim1;
	dsp_potential_dim1 = H5Screate(H5S_SIMPLE);
	int data_potential_dim1_RANK = 1;
	hsize_t data_potential_dim1_dim[] = {(long unsigned int)params->IM.m1};
	H5Sset_extent_simple(dsp_potential_dim1, data_potential_dim1_RANK, data_potential_dim1_dim, NULL);
	hid_t dst_potential_dim1;
	dst_potential_dim1 = H5Dcreate2(grp_potential_slice, "dim1", H5T_NATIVE_FLOAT, dsp_potential_dim1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(dst_potential_dim1, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, potential_x_coordinates);
	free(potential_x_coordinates);
 
//////////////////////////////  group "potential_slices" dataset "dim1" attribute "name"

	writeAttributeString(dst_potential_dim1, "name",  "x");

////////////////////////////// group "potential_slices" dataset "dim1" attribute "units"   
   
	writeAttributeString(dst_potential_dim1, "units",  "[m]");
	H5Sclose(dsp_potential_dim1); 
	H5Dclose(dst_potential_dim1); 
 
///////////////////////////// group "potential_slices" dataset "dim2" 
	
	float * potential_y_coordinates;
	potential_y_coordinates = ( float* ) malloc ( params->IM.m2 *  sizeof ( float ) );
	for (int i=0; i<params->IM.m2; i++)
	{
	  potential_y_coordinates[i] = i -(params->IM.m2-1)/2.0;
	}
    
	hid_t  dsp_potential_dim2;
	dsp_potential_dim2 = H5Screate(H5S_SIMPLE);
	int data_potential_dim2_RANK = 1;
	hsize_t data_potential_dim2_dim[] = {(long unsigned int)params->IM.m2};
	H5Sset_extent_simple(dsp_potential_dim2, data_potential_dim2_RANK, data_potential_dim2_dim, NULL);
	hid_t dst_potential_dim2;
	dst_potential_dim2 = H5Dcreate2(grp_potential_slice, "dim2", H5T_NATIVE_FLOAT, dsp_potential_dim2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(dst_potential_dim2, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, potential_y_coordinates);
	free(potential_y_coordinates);
////////////////////////////// group "potential_slices" dataset "dim2" attribute "name"
	
	writeAttributeString(dst_potential_dim2, "name",  "y");

////////////////////////////// group "potential_slices" dataset "dim2" attribute "units"   

	writeAttributeString(dst_potential_dim2, "units",  "[m]");

	H5Sclose(dsp_potential_dim2); 
	H5Dclose(dst_potential_dim2); 
    
///////////////////////////// group "potential_slices" dataset "dim3" 
    
	float * potential_z_coordinates;
	potential_z_coordinates = ( float* ) malloc ( params->IM.m3 *  sizeof ( float ) );
	for (int i=0; i<params->IM.m3; i++)
	{
	potential_z_coordinates[i] = i -(params->IM.m3-1)/2.0;
	}
   
	hid_t  dsp_potential_dim3;
	dsp_potential_dim3 = H5Screate(H5S_SIMPLE);
	int data_potential_dim3_RANK = 1;
	hsize_t data_potential_dim3_dim[] = {(long unsigned int)params->IM.m3};
	H5Sset_extent_simple(dsp_potential_dim3, data_potential_dim3_RANK, data_potential_dim3_dim, NULL);
	hid_t dst_potential_dim3;
	dst_potential_dim3 = H5Dcreate2(grp_potential_slice, "dim3", H5T_NATIVE_FLOAT, dsp_potential_dim3, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(dst_potential_dim3, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, potential_z_coordinates);
	free(potential_z_coordinates);
    
////////////////////////////// group "potential_slices" dataset "dim3" attribute "name"
  
	writeAttributeString(dst_potential_dim3, "name",  "z");
  
////////////////////////////// group "potential_slices" dataset "dim3" attribute "units"   

	writeAttributeString(dst_potential_dim3, "units",  "[m]");
   
	H5Sclose(dsp_potential_dim3); 
	H5Dclose(dst_potential_dim3); 
  
///////////////////////// group "potential_slices" dataset "dim4" 
      
	hid_t  dsp_potential_dim4;
	dsp_potential_dim4 = H5Screate(H5S_SIMPLE);
	int data_potential_dim4_RANK = 1;
	hsize_t data_potential_dim4_dim[] = {2};
	const char*  complex_name[2] = {"real", "imag"};
	hid_t dim4_data_type;
  	int slen = strlen (complex_name[0]);
 
	dim4_data_type = H5Tcopy(H5T_C_S1);
	H5Tset_size(dim4_data_type, slen);
	H5Tset_strpad(dim4_data_type,H5T_STR_NULLTERM);
	H5Sset_extent_simple(dsp_potential_dim4, data_potential_dim4_RANK, data_potential_dim4_dim, NULL);
	hid_t dst_potential_dim4;
	dst_potential_dim4 = H5Dcreate2(grp_potential_slice, "dim4", dim4_data_type, dsp_potential_dim4, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(dst_potential_dim4, dim4_data_type, H5S_ALL , H5S_ALL, H5P_DEFAULT, *complex_name);

////////////////////////////// group "potential_slices" dataset "dim4" attribute "name"
	
	hid_t dsp_potential_dim4_name;
	dsp_potential_dim4_name  = H5Screate(H5S_SCALAR);
	char strdim4[] = "complex";                /* Value of the string attribute */
	slen =strlen (strdim4);
	hid_t attrtype;
	hid_t attr_potential_dim4_name;
	attrtype = H5Tcopy(H5T_C_S1);
	H5Tset_size(attrtype, slen);
	H5Tset_strpad(attrtype,H5T_STR_NULLTERM);
	attr_potential_dim4_name = H5Acreate2(dst_potential_dim4, "name", attrtype, dsp_potential_dim4_name, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attr_potential_dim4_name, attrtype, strdim4);
	H5Sclose(dsp_potential_dim4_name);
	H5Aclose(attr_potential_dim4_name);

//////////////////////////////  group "potential_slices" dataset "dim4" attribute "units"   
   
	writeAttributeString(dst_potential_dim4, "units",  "[]");

	H5Sclose(dsp_potential_dim4); 
	H5Dclose(dst_potential_dim4); 
      
	H5Gclose(grp_potential_slice);
	}
   
	if (printLevel>1)
	{
///////////////////////////// subgroup of "data" exit_wave          
	 hid_t grp_exit_wave;
	  grp_exit_wave = H5Gcreate2(grp_data, "exit_wave", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
     
/////////////////////////// group "exit_wave" attribute "emd_group_type"    
    
	  unsigned char emd_exit_wave_type_value = 1;
	  writeAttributeSingle(grp_exit_wave,"emd_group_type",emd_exit_wave_type_value);  
   
/////////////////////////////// group "exit_wave" dataset "data"
 

   
   float * exitwave_xyz;
   exitwave_xyz = ( float* ) malloc ( params->IM.m1* params->IM.m2*  params->IM.n3 * 2*  sizeof ( float ) );
   
   for (int i= 0; i < params->IM.n3; i++)
      for (int j= 0; j <params->IM.m2; j++)
       for (int k= 0; k <params->IM.m1; k++)
	  {
	 exitwave_xyz[2*(k*params->IM.n3 *params->IM.m2 + j *params->IM.n3 + i) + 0 ] = exitwave[ 2 *(i*params->IM.m1 *params->IM.m2 + j *params->IM.m1 + k) + 0  ];
         exitwave_xyz[2*(k*params->IM.n3 *params->IM.m2 + j *params->IM.n3 + i) + 1 ] = exitwave[ 2 *(i*params->IM.m1 *params->IM.m2 + j *params->IM.m1 + k) + 1  ];
	  }
    
   hid_t  dsp_exit_wave_data;
   dsp_exit_wave_data = H5Screate(H5S_SIMPLE);
   int data_exit_wave_RANK = 4;
   hsize_t data_exit_wave_dim[] = {(long unsigned int)params->IM.m1, (long unsigned int)params->IM.m2, (long unsigned int) params->IM.n3, 2};
   H5Sset_extent_simple(dsp_exit_wave_data, data_exit_wave_RANK, data_exit_wave_dim, NULL);
   hid_t dst_exit_wave_data;
   dst_exit_wave_data = H5Dcreate2(grp_exit_wave, "data", H5T_NATIVE_FLOAT, dsp_exit_wave_data, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_exit_wave_data, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, exitwave_xyz);
   H5Sclose(dsp_exit_wave_data); 
   H5Dclose(dst_exit_wave_data);
 
   /////////////////////// group "exit_wave" dataset "dim1" 
   
   float * exit_wave_x_coordinates;
   exit_wave_x_coordinates = ( float* ) malloc ( params->IM.m1 *  sizeof ( float ) );
   for (int i=0; i<params->IM.m1; i++)
   {
     exit_wave_x_coordinates[i] = i -(params->IM.m1-1)/2.0;
   }
    
   hid_t  dsp_exit_wave_dim1;
   dsp_exit_wave_dim1 = H5Screate(H5S_SIMPLE);
   int data_exit_wave_dim1_RANK = 1;
   hsize_t data_exit_wave_dim1_dim[] = {(long unsigned int)params->IM.m1};
   H5Sset_extent_simple(dsp_exit_wave_dim1, data_exit_wave_dim1_RANK, data_exit_wave_dim1_dim, NULL);
   hid_t dst_exit_wave_dim1;
   dst_exit_wave_dim1 = H5Dcreate2(grp_exit_wave, "dim1", H5T_NATIVE_FLOAT, dsp_exit_wave_dim1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_exit_wave_dim1, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, exit_wave_x_coordinates);
   free(exit_wave_x_coordinates);
 
//////////////////////////////  group "potential_slices" dataset "dim1" attribute "name"
   
   writeAttributeString(dst_exit_wave_dim1, "name",  "x");

////////////////////////////// group "potential_slices" dataset "dim1" attribute "units"   
   
   writeAttributeString(dst_exit_wave_dim1, "units",  "[m]");
   
   H5Sclose(dsp_exit_wave_dim1); 
   H5Dclose(dst_exit_wave_dim1); 

///////////////////////////// group "potential_slices" dataset "dim2" 
   float * exit_wave_y_coordinates;
   exit_wave_y_coordinates = ( float* ) malloc ( params->IM.m2 *  sizeof ( float ) );
   for (int i=0; i<params->IM.m2; i++)
   {
     exit_wave_y_coordinates[i] = i -(params->IM.m2-1)/2.0;
   }
    
   hid_t  dsp_exit_wave_dim2;
   dsp_exit_wave_dim2 = H5Screate(H5S_SIMPLE);
   int data_exit_wave_dim2_RANK = 1;
   hsize_t data_exit_wave_dim2_dim[] = {(long unsigned int)params->IM.m2};
   H5Sset_extent_simple(dsp_exit_wave_dim2, data_exit_wave_dim2_RANK, data_exit_wave_dim2_dim, NULL);
   hid_t dst_exit_wave_dim2;
   dst_exit_wave_dim2 = H5Dcreate2(grp_exit_wave, "dim2", H5T_NATIVE_FLOAT, dsp_exit_wave_dim2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_exit_wave_dim2, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, exit_wave_y_coordinates);
   free(exit_wave_y_coordinates);
  
////////////////////////////// group "exit_wave" dataset "dim2" attribute "name"

   writeAttributeString(dst_exit_wave_dim2, "name",  "y");
 
////////////////////////////// group "exit_wave" dataset "dim2" attribute "units"   

   writeAttributeString(dst_exit_wave_dim2, "units",  "[m]");
   
   H5Sclose(dsp_exit_wave_dim2); 
   H5Dclose(dst_exit_wave_dim2); 
    
///////////////////////////// group "exit_wave" dataset "dim3" 
    
   float * exit_wave_z_index;
   exit_wave_z_index = ( float* ) malloc ( params->IM.n3 *  sizeof ( float ) );
   for (int i=0; i<params->IM.n3; i++)
   {
     exit_wave_z_index[i] = (float) i;
   }
   
   hid_t  dsp_exit_wave_dim3;
   dsp_exit_wave_dim3 = H5Screate(H5S_SIMPLE);
   int data_exit_wave_dim3_RANK = 1;
   hsize_t data_exit_wave_dim3_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_exit_wave_dim3, data_exit_wave_dim3_RANK, data_exit_wave_dim3_dim, NULL);
   hid_t dst_exit_wave_dim3;
   dst_exit_wave_dim3 = H5Dcreate2(grp_exit_wave, "dim3", H5T_NATIVE_FLOAT, dsp_exit_wave_dim3, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_exit_wave_dim3, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, exit_wave_z_index);
   free(exit_wave_z_index);
  
////////////////////////////// group "exit_wave" dataset "dim3" attribute "name"

   writeAttributeString(dst_exit_wave_dim3, "name",  "z");
 
////////////////////////////// group "exit_wave" dataset "dim3" attribute "units"   
   
   writeAttributeString(dst_exit_wave_dim3, "units",  "[m]");
    
   H5Sclose(dsp_exit_wave_dim3); 
   H5Dclose(dst_exit_wave_dim3); 
  
// ///////////////////////// group "exit_wave" dataset "dim4" 
   hid_t  dsp_exit_wave_dim4;
   dsp_exit_wave_dim4 = H5Screate(H5S_SIMPLE);
   int data_exit_wave_dim4_RANK = 1;
   hsize_t data_exit_wave_dim4_dim[] = {2};
   const char*  complex_name[2] = {"real", "imag"};

   hid_t dim4_data_type;
   int slen = strlen (complex_name[0]);
   dim4_data_type = H5Tcopy(H5T_C_S1);
   H5Tset_size(dim4_data_type, slen);
   H5Tset_strpad(dim4_data_type,H5T_STR_NULLTERM);
   H5Sset_extent_simple(dsp_exit_wave_dim4, data_exit_wave_dim4_RANK, data_exit_wave_dim4_dim, NULL);
   hid_t dst_exit_wave_dim4;
   dst_exit_wave_dim4 = H5Dcreate2(grp_exit_wave, "dim4", dim4_data_type, dsp_exit_wave_dim4, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_exit_wave_dim4, dim4_data_type, H5S_ALL , H5S_ALL, H5P_DEFAULT, *complex_name);
  
////////////////////////////// group "exit_wave" dataset "dim4" attribute "name"
   hid_t dsp_exit_wave_dim4_name;
   dsp_exit_wave_dim4_name  = H5Screate(H5S_SCALAR);
   char strdim4[] = "complex";                /* Value of the string attribute */
   slen =strlen (strdim4);
   hid_t attrtype;
   hid_t attr_exit_wave_dim4_name;
   attrtype = H5Tcopy(H5T_C_S1);
   H5Tset_size(attrtype, slen);
   H5Tset_strpad(attrtype,H5T_STR_NULLTERM);
   attr_exit_wave_dim4_name = H5Acreate2(dst_exit_wave_dim4, "name", attrtype, dsp_exit_wave_dim4_name, H5P_DEFAULT, H5P_DEFAULT);
   H5Awrite(attr_exit_wave_dim4_name, attrtype, strdim4);
   H5Sclose(dsp_exit_wave_dim4_name);
   H5Aclose(attr_exit_wave_dim4_name);

//////////////////////////////  group "exit_wave_slices" dataset "dim4" attribute "units"   

   writeAttributeString(dst_exit_wave_dim4, "units",  "[]");

   H5Sclose(dsp_exit_wave_dim4); 
   H5Dclose(dst_exit_wave_dim4); 
   H5Gclose(grp_exit_wave);
   }
   
//////////////////////////////  subgroup of "data" images
  
   hid_t grp_images;
   grp_images = H5Gcreate2(grp_data, "images", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   
/////////////////////////////// group "images" attribute "emd_group_type"    
  
   unsigned char emd_images_type_value = 1;
   writeAttributeSingle(grp_images,"emd_group_type",emd_images_type_value);  
   
/////////////////////////////// group "images" dataset "data"
   float * f_xyz;
   f_xyz = ( float* ) malloc ( params->IM.n1 *params->IM.n2 *params->IM.n3 *  sizeof ( float ) );
   
   for (int i= 0; i <params->IM.n3; i++)
      for (int j= 0; j <params->IM.n2; j++)
       for (int k= 0; k <params->IM.n1; k++)
       {
	 f_xyz[k*params->IM.n3 *params->IM.n2 + j *params->IM.n3 + i ] =  image[i*params->IM.n1 *params->IM.n2 + j *params->IM.n1 + k  ];
       }
   
   hid_t  dsp_images_data;
   dsp_images_data = H5Screate(H5S_SIMPLE);
   int data_images_RANK = 3;
   hsize_t data_images_dim[] = {(long unsigned int)params->IM.n1, (long unsigned int)params->IM.n2,(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_images_data, data_images_RANK, data_images_dim, NULL);
   hid_t dst_images_data;
   dst_images_data = H5Dcreate2(grp_images, "data", H5T_NATIVE_FLOAT, dsp_images_data, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_images_data, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, f_xyz);
   H5Sclose(dsp_images_data); 
   H5Dclose(dst_images_data);
   
/////////////////////////////////////// group "images" dataset "dim1" 
   float * images_x_coordinates;
   images_x_coordinates = ( float* ) malloc ( params->IM.n1 *  sizeof ( float ) );
   for (int i=0; i<params->IM.n1; i++)
   {
     images_x_coordinates[i] = i -(params->IM.n1-1)/2.0;
   }
  
   hid_t  dsp_images_dim1;
   dsp_images_dim1 = H5Screate(H5S_SIMPLE);
   int data_images_dim1_RANK = 1;
   hsize_t data_images_dim1_dim[] = {(long unsigned int)params->IM.n1};
   H5Sset_extent_simple(dsp_images_dim1, data_images_dim1_RANK, data_images_dim1_dim, NULL);
   hid_t dst_images_dim1;
   dst_images_dim1 = H5Dcreate2(grp_images, "dim1", H5T_NATIVE_FLOAT, dsp_images_dim1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_images_dim1, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, images_x_coordinates);
   free(images_x_coordinates);
   

////////////////////////////// group "images" dataset "dim2" attribute "name"
  
   writeAttributeString(dst_images_dim1, "name",  "x");
 
//////////////////////////////  group "images" dataset "dim2" attribute "units"   
   
   writeAttributeString(dst_images_dim1, "units",  "[m]");
   
   H5Sclose(dsp_images_dim1); 
   H5Dclose(dst_images_dim1); 
   
///////////////////////////// group "images" dataset "dim2" 
   
   float * images_y_coordinates;
   images_y_coordinates = ( float* ) malloc ( params->IM.n2 *  sizeof ( float ) );
   for (int i=0; i<params->IM.n2; i++)
   {
     images_y_coordinates[i] = i -(params->IM.n2-1)/2.0;
   }
   
   hid_t  dsp_images_dim2;
   dsp_images_dim2 = H5Screate(H5S_SIMPLE);
   int data_images_dim2_RANK = 1;
   hsize_t data_images_dim2_dim[] = {(long unsigned int)params->IM.n2};
   H5Sset_extent_simple(dsp_images_dim2, data_images_dim2_RANK, data_images_dim2_dim, NULL);
   hid_t dst_images_dim2;
   dst_images_dim2 = H5Dcreate2(grp_images, "dim2", H5T_NATIVE_FLOAT, dsp_images_dim2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_images_dim2, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, images_y_coordinates);
   free(images_y_coordinates);
   
////////////////////////////// group "images" dataset "dim2" attribute "name"

   writeAttributeString(dst_images_dim2, "name",  "y");
 
//////////////////////////////  group "images" dataset "dim2" attribute "units"   

   writeAttributeString(dst_images_dim2, "units",  "[m]");
   
   H5Sclose(dsp_images_dim2); 
   H5Dclose(dst_images_dim2); 
   
   ///////////////////////// group "images" dataset "dim3" 

   float * images_z_index;
   images_z_index = ( float* ) malloc ( params->IM.n3 *  sizeof ( float ) );
   for (int i=0; i<params->IM.n3; i++)
   {
     images_z_index[i] = (float) i;
   }
   
   hid_t  dsp_images_dim3;
   dsp_images_dim3 = H5Screate(H5S_SIMPLE);
   int data_images_dim3_RANK = 1;
   hsize_t data_images_dim3_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_images_dim3, data_images_dim3_RANK, data_images_dim3_dim, NULL);
   hid_t dst_images_dim3;
   dst_images_dim3 = H5Dcreate2(grp_images, "dim3", H5T_NATIVE_FLOAT, dsp_images_dim3, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_images_dim3, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, images_z_index);
   free(images_z_index);
  
////////////////////////////// group "images" dataset "dim3" attribute "name"
   writeAttributeString(dst_images_dim3, "name",  "z");
// 
////////////////////////////// group "images" dataset "dim3" attribute "units"   
   
   writeAttributeString(dst_images_dim3, "units",  "[m]");
 
   H5Sclose(dsp_images_dim3); 
   H5Dclose(dst_images_dim3); 
    
   H5Gclose(grp_images);
   H5Gclose(grp_data);
   
////////////////////////////////  group "microscope" 
   
   hid_t grp_microscope;
   grp_microscope = H5Gcreate2(file, "/microscope", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

////////////////////////////////  group "microscope" attribute "voltage"
   
   writeAttributeSingle(grp_microscope,"voltage",params->EM.E0);  
   
////////////////////////////////  group "microscope" attribute "voltage_units"   
   writeAttributeString(grp_microscope, "voltage_units",  "[v]");
   
////////////////////////////////  group "microscope" attribute "gamma"
   
   writeAttributeSingle(grp_microscope,"gamma",params->EM.gamma);  
   
////////////////////////////////  group "microscope" attribute "wavelength"
   
   writeAttributeSingle(grp_microscope,"wavelength",params->EM.lambda);  
   
////////////////////////////////  group "microscope" attribute "wavelength_units"
   
   writeAttributeString(grp_microscope, "wavelength_units",  "[m]");

////////////////////////////////  group "microscope" attribute "interaction_constant"

   writeAttributeSingle(grp_microscope,"interaction_constant",params->EM.sigma);  

////////////////////////////////  group "microscope" attribute "interaction_constant_units"
   writeAttributeString(grp_microscope, "interaction_constant_units",  "[V^-1][m^-1]");
   
////////////////////////////////  group "microscope" attribute "focus_spread"
   writeAttributeSingle(grp_microscope,"focus_spread",params->EM.defocspread);
   
////////////////////////////////  group "microscope" attribute "focus_spread_units"
   writeAttributeString(grp_microscope, "focus_spread_units",  "[m]");
   
////////////////////////////////  group "microscope" attribute "illumination_angle"
   writeAttributeSingle(grp_microscope,"illumination_angle",params->EM.illangle);

////////////////////////////////  group "microscope" attribute "illumination_angle_units"
   writeAttributeString(grp_microscope, "illumination_angle_units",  "[rad]");
   
////////////////////////////////  group "microscope" attribute "objective_aperture"
   writeAttributeSingle(grp_microscope,"objective_aperture",params->EM.ObjAp);
   
////////////////////////////////  group "microscope" attribute "objective_aperture_units"
   writeAttributeString(grp_microscope, "objective_aperture_units",  "[rad]");
   
////////////////////////////////  group "microscope" attribute "mtf_a"
   writeAttributeSingle(grp_microscope,"mtf_a",params->EM.mtfa);

////////////////////////////////  group "microscope" attribute "mtf_b"
   writeAttributeSingle(grp_microscope,"mtf_b",params->EM.mtfb);

////////////////////////////////  group "microscope" attribute "mtf_c"
   writeAttributeSingle(grp_microscope,"mtf_c",params->EM.mtfc);

////////////////////////////////  group "microscope" attribute "mtf_d"   
   writeAttributeSingle(grp_microscope,"mtf_d",params->EM.mtfd);
   
////////////////////////////////  group "microscope" group "aberrations"   
   hid_t grp_aberrations;
   
   grp_aberrations = H5Gcreate2(grp_microscope, "aberrations", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "amplitude_units"   
   
   writeAttributeString(grp_aberrations, "amplitude_units",  "[m]");
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "amplitude_units" 
   
   writeAttributeString(grp_aberrations, "angle_units",  "[rad]");

////////////////////////////////  group "microscope" group "aberrations"  attribute "C1_amplitude" 
   
   writeAttributeSingle(grp_aberrations,"C1_amplitude",params->EM.aberration.C1_0);

////////////////////////////////  group "microscope" group "aberrations"  attribute "C1_angle" 
   
//    writeAttributeSingle(grp_aberrations,"C1_angle",params->EM.aberration.C1_1);

////////////////////////////////  group "microscope" group "aberrations"  attribute "A1_amplitude" 
   
   writeAttributeSingle(grp_aberrations,"A1_amplitude",params->EM.aberration.A1_0);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A1_angle" 
   
   writeAttributeSingle(grp_aberrations,"A1_angle",params->EM.aberration.A1_1);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A2_amplitude" 
   
   writeAttributeSingle(grp_aberrations,"A2_amplitude",params->EM.aberration.A2_0);

////////////////////////////////  group "microscope" group "aberrations"  attribute "A2_angle" 
   
   writeAttributeSingle(grp_aberrations,"A2_angle",params->EM.aberration.A2_1);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "B2_amplitude"
   
   writeAttributeSingle(grp_aberrations,"B2_amplitude",params->EM.aberration.B2_0);

////////////////////////////////  group "microscope" group "aberrations"  attribute "B2_angle" 
   
   writeAttributeSingle(grp_aberrations,"B2_angle",params->EM.aberration.B2_1);   

////////////////////////////////  group "microscope" group "aberrations"  attribute "C3_amplitude" 
   
   writeAttributeSingle(grp_aberrations,"C3_amplitude",params->EM.aberration.C3_0);  

////////////////////////////////  group "microscope" group "aberrations"  attribute "A3_amplitude" 
   
   writeAttributeSingle(grp_aberrations,"A3_amplitude",params->EM.aberration.A3_0);     
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A3_angle" 
   
   writeAttributeSingle(grp_aberrations,"A3_angle",params->EM.aberration.A3_1);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S3_amplitude" 
   
   writeAttributeSingle(grp_aberrations,"S3_amplitude",params->EM.aberration.S3_0);        
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S3_angle" 
   
   writeAttributeSingle(grp_aberrations,"S3_angle",params->EM.aberration.S3_1);   
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A4_amplitude" 
   
   writeAttributeSingle(grp_aberrations,"A4_amplitude",params->EM.aberration.A4_0); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A4_angle" 
   
   writeAttributeSingle(grp_aberrations,"A4_angle",params->EM.aberration.A4_1);

////////////////////////////////  group "microscope" group "aberrations"  attribute "B4_amplitude" 
   
   writeAttributeSingle(grp_aberrations,"B4_amplitude",params->EM.aberration.B4_0); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "B4_angle" 
   
   writeAttributeSingle(grp_aberrations,"B4_angle",params->EM.aberration.B4_1);
 
////////////////////////////////  group "microscope" group "aberrations"  attribute "D4_amplitude" 
   
   writeAttributeSingle(grp_aberrations,"D4_amplitude",params->EM.aberration.D4_0);   
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "D4_angle" 
   
   writeAttributeSingle(grp_aberrations,"D4_angle",params->EM.aberration.D4_1); 

////////////////////////////////  group "microscope" group "aberrations"  attribute "C5_amplitude" 
   writeAttributeSingle(grp_aberrations,"C5_amplitude",params->EM.aberration.C5_0);  
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "C5_angle" 
//    writeAttributeSingle(grp_aberrations,"C5_angle",params->EM.aberration.C5_1);   

////////////////////////////////  group "microscope" group "aberrations"  attribute "A5_amplitude" 
   writeAttributeSingle(grp_aberrations,"A5_amplitude",params->EM.aberration.A5_0);      

////////////////////////////////  group "microscope" group "aberrations"  attribute "A5_angle" 
   writeAttributeSingle(grp_aberrations,"A5_angle",params->EM.aberration.A5_1); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "R5_amplitude" 
   writeAttributeSingle(grp_aberrations,"R5_amplitude",params->EM.aberration.R5_0);  
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "R5_angle" 
   writeAttributeSingle(grp_aberrations,"R5_angle",params->EM.aberration.R5_1); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S5_amplitude" 
   writeAttributeSingle(grp_aberrations,"S5_amplitude",params->EM.aberration.S5_0);  
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S5_angle" 
   writeAttributeSingle(grp_aberrations,"S5_angle",params->EM.aberration.S5_1);    
   
   H5Gclose(grp_aberrations);
   H5Gclose(grp_microscope);
   
////////////////////////////////  group "sample"    
   
   hid_t grp_sample;
   grp_sample = H5Gcreate2(file, "/sample", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

////////////////////////////////  group "sample" attribute "name"      
   
   writeAttributeString(grp_sample, "name",  params->SAMPLE.sample_name);

////////////////////////////////  group "sample" attribute "material"      
   
   writeAttributeString(grp_sample, "material",  params->SAMPLE.material);   
  
// ///////////////////////////////// group "sample" dataset "atomic_numbers" 
   
   hid_t  dsp_sample_atomic_numbers;
   dsp_sample_atomic_numbers = H5Screate(H5S_SIMPLE);
   int data_sample_atomic_numbers_RANK = 1;
   hsize_t data_sample_atomic_numbers_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_atomic_numbers, data_sample_atomic_numbers_RANK, data_sample_atomic_numbers_dim, NULL);
   hid_t dst_sample_atomic_numbers;
   dst_sample_atomic_numbers = H5Dcreate2(grp_sample, "atomic_numbers", H5T_NATIVE_INT, dsp_sample_atomic_numbers, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_atomic_numbers, H5T_NATIVE_INT, H5S_ALL , H5S_ALL, H5P_DEFAULT, Z_h);   
   H5Dclose(dst_sample_atomic_numbers);
   H5Sclose(dsp_sample_atomic_numbers); 

///////////////////////////////// group "sample" dataset "x_coordinates"
   
   hid_t  dsp_sample_x_coordinates;
   dsp_sample_x_coordinates = H5Screate(H5S_SIMPLE);
   int data_sample_x_coordinates_RANK = 1;
   hsize_t data_sample_x_coordinates_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_x_coordinates, data_sample_x_coordinates_RANK, data_sample_x_coordinates_dim, NULL);
   hid_t dst_sample_x_coordinates;
   dst_sample_x_coordinates = H5Dcreate2(grp_sample, "x_coordinates", H5T_NATIVE_FLOAT, dsp_sample_x_coordinates, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_x_coordinates, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, xCoord_h);   
   
///////////////////////////////// group "sample" dataset "x_coordinates" attribute "units"   
   
   writeAttributeString(dst_sample_x_coordinates, "units",  "[m]");
   
   H5Sclose(dsp_sample_x_coordinates);
   H5Dclose(dst_sample_x_coordinates);   
//    
///////////////////////////////// group "sample" dataset "y_coordinates"
   hid_t  dsp_sample_y_coordinates;
   dsp_sample_y_coordinates = H5Screate(H5S_SIMPLE);
   int data_sample_y_coordinates_RANK = 1;
   hsize_t data_sample_y_coordinates_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_y_coordinates, data_sample_y_coordinates_RANK, data_sample_y_coordinates_dim, NULL);
   hid_t dst_sample_y_coordinates;
   dst_sample_y_coordinates = H5Dcreate2(grp_sample, "y_coordinates", H5T_NATIVE_FLOAT, dsp_sample_y_coordinates, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_y_coordinates, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, yCoord_h);   

///////////////////////////////// group "sample" dataset "y_coordinates" attribute "units"   
   writeAttributeString(dst_sample_y_coordinates, "units",  "[m]");

   H5Sclose(dsp_sample_y_coordinates);
   H5Dclose(dst_sample_y_coordinates);  
// 
///////////////////////////////// group "sample" dataset "z_coordinates"
   hid_t  dsp_sample_z_coordinates;
   dsp_sample_z_coordinates = H5Screate(H5S_SIMPLE);
   int data_sample_z_coordinates_RANK = 1;
   hsize_t data_sample_z_coordinates_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_z_coordinates, data_sample_z_coordinates_RANK, data_sample_z_coordinates_dim, NULL);
   hid_t dst_sample_z_coordinates;
   dst_sample_z_coordinates = H5Dcreate2(grp_sample, "z_coordinates", H5T_NATIVE_FLOAT, dsp_sample_z_coordinates, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_z_coordinates, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, zCoord_h);   

///////////////////////////////// group "sample" dataset "z_coordinates" attribute "units"   
   writeAttributeString(dst_sample_z_coordinates, "units",  "[m]");

   H5Sclose(dsp_sample_z_coordinates);
   H5Dclose(dst_sample_z_coordinates);   

///////////////////////////////// group "sample" dataset "debeye_waller_factors "
   hid_t  dsp_sample_debeye_waller_factors;
   dsp_sample_debeye_waller_factors = H5Screate(H5S_SIMPLE);
   int data_sample_debeye_waller_factors_RANK = 1;
   hsize_t data_sample_debeye_waller_factors_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_debeye_waller_factors, data_sample_debeye_waller_factors_RANK, data_sample_debeye_waller_factors_dim, NULL);
   hid_t dst_sample_debeye_waller_factors;
   dst_sample_debeye_waller_factors = H5Dcreate2(grp_sample, "debeye_waller_factors", H5T_NATIVE_FLOAT, dsp_sample_debeye_waller_factors, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_debeye_waller_factors, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, DWF_h);   

///////////////////////////////// group "sample" dataset "debeye_waller_factors" attribute "units"   
   writeAttributeString(dst_sample_debeye_waller_factors, "units",  "[m^2]");

   H5Sclose(dsp_sample_debeye_waller_factors);
   H5Dclose(dst_sample_debeye_waller_factors);   
   
///////////////////////////////// group "sample" dataset "occupancy"
   hid_t  dsp_sample_occupancy;
   dsp_sample_occupancy = H5Screate(H5S_SIMPLE);
   int data_sample_occupancy_RANK = 1;
   hsize_t data_sample_occupancy_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_occupancy, data_sample_occupancy_RANK, data_sample_occupancy_dim, NULL);
   hid_t dst_sample_occupancy;
   dst_sample_occupancy = H5Dcreate2(grp_sample, "occupancy", H5T_NATIVE_FLOAT, dsp_sample_occupancy, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_occupancy, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, occ_h);   

   H5Sclose(dsp_sample_occupancy);
   H5Dclose(dst_sample_occupancy);      
//    
/////////////////////////////// group "sample" attribute "absorptive_potential_factor"   /// ??
   writeAttributeSingle(grp_sample,"absorptive_potential_factor",params->SAMPLE.imPot);    
      
    
   H5Gclose(grp_sample);
    

////////////////////////////////  group "imaging"         
    hid_t grp_imaging;
    grp_imaging = H5Gcreate2(file, "/imaging", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

///////////////////////////////// group "imaging" attribute "mode"      

    writeAttributeSingle(grp_imaging,"mode",params->IM.mode);  

///////////////////////////////// group "imaging" attribute "sample_size_x"  
    
    writeAttributeSingle(grp_imaging,"sample_size_x",params->IM.m1);        

///////////////////////////////// group "imaging" attribute "sample_size_y"  
    
    writeAttributeSingle(grp_imaging,"sample_size_y",params->IM.m2);       

///////////////////////////////// group "imaging" attribute "sample_size_z"  
    
    writeAttributeSingle(grp_imaging,"sample_size_z",params->IM.m3);       

///////////////////////////////// group "imaging" attribute "sample_size_units" 
    
    writeAttributeString(grp_imaging, "sample_size_units",  "[pix]");   
//     
///////////////////////////////// group "imaging" attribute "pixel_size_x"  
    
    writeAttributeSingle(grp_imaging,"pixel_size_x",params->IM.d1);     
    
///////////////////////////////// group "imaging" attribute "pixel_size_y"  
    
    writeAttributeSingle(grp_imaging,"pixel_size_y",params->IM.d2);       
    
///////////////////////////////// group "imaging" attribute "pixel_size_z"  
    
    writeAttributeSingle(grp_imaging,"pixel_size_z",params->IM.d3);        
    
///////////////////////////////// group "imaging" attribute "sample_size_units"  
    
    writeAttributeString(grp_imaging, "pixel_size_units",  "[m]");     
    
///////////////////////////////// group "imaging" attribute "image_size_x"  
    
    writeAttributeSingle(grp_imaging,"image_size_x",params->IM.n1);     
    
///////////////////////////////// group "imaging" attribute "image_size_y"  
    
    writeAttributeSingle(grp_imaging,"image_size_y",params->IM.n2);       
    
///////////////////////////////// group "imaging" attribute "image_size_z"  
    
    writeAttributeSingle(grp_imaging,"image_size_z",params->IM.n3);    
    
///////////////////////////////// group "imaging" attribute "image_size_units"  
    
    writeAttributeString(grp_imaging, "image_size_units",  "[pix]");      
//     
///////////////////////////////// group "imaging" attribute "border_size_x"  
    
    writeAttributeSingle(grp_imaging,"border_size_x",params->IM.dn1);     
    
///////////////////////////////// group "imaging" attribute "border_size_y"  
    
    writeAttributeSingle(grp_imaging,"border_size_y",params->IM.dn2);   
    
///////////////////////////////// group "imaging" attribute "border_size_units"  
    
    writeAttributeString(grp_imaging, "border_size_units",  "[pix]");      
// //     
///////////////////////////////// group "imaging" attribute "specimen_tilt_offset_x"  
    
    writeAttributeSingle(grp_imaging,"specimen_tilt_offset_x",params->IM.specimen_tilt_offset_x); 
    
    
///////////////////////////////// group "imaging" attribute "specimen_tilt_offset_y"  
    writeAttributeSingle(grp_imaging,"specimen_tilt_offset_y",params->IM.specimen_tilt_offset_y);  
    
    
///////////////////////////////// group "imaging" attribute "specimen_tilt_offset_z"  
    writeAttributeSingle(grp_imaging,"specimen_tilt_offset_z",params->IM.specimen_tilt_offset_z);  
    
    
///////////////////////////////// group "imaging" attribute "image_size_units"  
    
    writeAttributeString(grp_imaging, "specimen_tilt_offset_units",  "[rad]");   
    
    
///////////////////////////////// group "imaging" attribute "frozen_phonons"  
    
    writeAttributeSingle(grp_imaging,"frozen_phonons",params->IM.frPh);     
    
///////////////////////////////// group "imaging" attribute "pixel_dose"  
    
    writeAttributeSingle(grp_imaging,"pixel_dose",params->IM.pD);       
    
///////////////////////////////// group "imaging" attribute "subpixel_size_z"  
    
    writeAttributeSingle(grp_imaging,"subpixel_size_z",params->IM.subSlTh);
    
///////////////////////////////// group "imaging" attribute "image_size_units"  
    
    writeAttributeString(grp_imaging, "subpixel_size_units",  "[m]");   
    
      
    float *specimen_tilt_x, *specimen_tilt_y;
    specimen_tilt_x = ( float* ) malloc (  params->IM.n3*  sizeof ( float ) );
    specimen_tilt_y = ( float* ) malloc (  params->IM.n3*  sizeof ( float ) );
    for(int i=0; i< params->IM.n3; i++ )
    {
      specimen_tilt_x[i] = params->IM.tiltspec[2*i+0];
      specimen_tilt_y[i] = params->IM.tiltspec[2*i+1];
    }
    
// ///////////////////////////////// group "imaging" dataset "specimen_tilt_x" 
   hid_t  dsp_imaging_specimen_tilt_x;
   dsp_imaging_specimen_tilt_x = H5Screate(H5S_SIMPLE);
   int data_imaging_specimen_tilt_x_RANK = 1;
   hsize_t data_imaging_specimen_tilt_x_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_imaging_specimen_tilt_x, data_imaging_specimen_tilt_x_RANK, data_imaging_specimen_tilt_x_dim, NULL);
   hid_t dst_imaging_specimen_tilt_x;
   dst_imaging_specimen_tilt_x = H5Dcreate2(grp_imaging, "specimen_tilt_x", H5T_NATIVE_FLOAT, dsp_imaging_specimen_tilt_x, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_imaging_specimen_tilt_x, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, specimen_tilt_x); 
   

   
///////////////////////////////// group "imaging" dataset "specimen_tilt_x"  attribute "units"  
   writeAttributeString(dst_imaging_specimen_tilt_x, "units",  "[rad]");   
    
   H5Dclose(dst_imaging_specimen_tilt_x);
   H5Sclose(dsp_imaging_specimen_tilt_x);    
    
// ///////////////////////////////// group "imaging" dataset "specimen_tilt_y" 
   hid_t  dsp_imaging_specimen_tilt_y;
   dsp_imaging_specimen_tilt_y = H5Screate(H5S_SIMPLE);
   int data_imaging_specimen_tilt_y_RANK = 1;
   hsize_t data_imaging_specimen_tilt_y_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_imaging_specimen_tilt_y, data_imaging_specimen_tilt_y_RANK, data_imaging_specimen_tilt_y_dim, NULL);
   hid_t dst_imaging_specimen_tilt_y;
   dst_imaging_specimen_tilt_y = H5Dcreate2(grp_imaging, "specimen_tilt_y", H5T_NATIVE_FLOAT, dsp_imaging_specimen_tilt_y, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_imaging_specimen_tilt_y, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, specimen_tilt_y); 
   
///////////////////////////////// group "imaging" dataset "specimen_tilt_y"  attribute "units"  
   writeAttributeString(dst_imaging_specimen_tilt_y, "units",  "[rad]");   
    
   H5Dclose(dst_imaging_specimen_tilt_y);
   H5Sclose(dsp_imaging_specimen_tilt_y);   
   
   
    float *beam_tilt_x, *beam_tilt_y;
    beam_tilt_x = ( float* ) malloc (  params->IM.n3*  sizeof ( float ) );
    beam_tilt_y = ( float* ) malloc (  params->IM.n3*  sizeof ( float ) );
    
    for(int i=0; i< params->IM.n3; i++ )
    {
      beam_tilt_x[i] = params->IM.tiltbeam[2*i+0];
      beam_tilt_y[i] = params->IM.tiltbeam[2*i+1];
    }
// ///////////////////////////////// group "imaging" dataset "beam_tilt_x" 
   hid_t  dsp_imaging_beam_tilt_x;
   dsp_imaging_beam_tilt_x = H5Screate(H5S_SIMPLE);
   int data_imaging_beam_tilt_x_RANK = 1;
   hsize_t data_imaging_beam_tilt_x_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_imaging_beam_tilt_x, data_imaging_beam_tilt_x_RANK, data_imaging_beam_tilt_x_dim, NULL);
   hid_t dst_imaging_beam_tilt_x;
   dst_imaging_beam_tilt_x = H5Dcreate2(grp_imaging, "beam_tilt_x", H5T_NATIVE_FLOAT, dsp_imaging_beam_tilt_x, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_imaging_beam_tilt_x, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, beam_tilt_x); 
   
///////////////////////////////// group "imaging" dataset "beam_tilt_x"  attribute "units"  
    writeAttributeString(dst_imaging_beam_tilt_x, "units",  "[rad]");   
    
   H5Dclose(dst_imaging_beam_tilt_x);
   H5Sclose(dsp_imaging_beam_tilt_x); 
// 
// ///////////////////////////////// group "imaging" dataset "beam_tilt_y" 
   hid_t  dsp_imaging_beam_tilt_y;
   dsp_imaging_beam_tilt_y = H5Screate(H5S_SIMPLE);
   int data_imaging_beam_tilt_y_RANK = 1;
   hsize_t data_imaging_beam_tilt_y_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_imaging_beam_tilt_y, data_imaging_beam_tilt_y_RANK, data_imaging_beam_tilt_y_dim, NULL);
   hid_t dst_imaging_beam_tilt_y;
   dst_imaging_beam_tilt_y = H5Dcreate2(grp_imaging, "beam_tilt_y", H5T_NATIVE_FLOAT, dsp_imaging_beam_tilt_y, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_imaging_beam_tilt_y, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, beam_tilt_y); 

///////////////////////////////// group "imaging" dataset "beam_tilt_y"  attribute "units"  
 
   writeAttributeString(dst_imaging_beam_tilt_y, "units",  "[rad]");   
    
   H5Dclose(dst_imaging_beam_tilt_y);
   H5Sclose(dsp_imaging_beam_tilt_y); 

   
///////////////////////////////// group "imaging" dataset "defoci" 
  
   hid_t  dsp_imaging_defoci;
   dsp_imaging_defoci = H5Screate(H5S_SIMPLE);
   int data_imaging_defoci_RANK = 1;
   hsize_t data_imaging_defoci_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_imaging_defoci, data_imaging_defoci_RANK, data_imaging_defoci_dim, NULL);
   hid_t dst_imaging_defoci;
   dst_imaging_defoci = H5Dcreate2(grp_imaging, "defoci", H5T_NATIVE_FLOAT, dsp_imaging_defoci, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_imaging_defoci, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, params->IM.defoci); 
   
///////////////////////////////// group "imaging" dataset "defoci"  attribute "units"  

   writeAttributeString(dst_imaging_defoci, "units",  "[rad]");   
  
   H5Dclose(dst_imaging_defoci);
   H5Sclose(dsp_imaging_defoci); 
   
    H5Gclose(grp_imaging);  


////////////////////////////////  group "user"       
    
    hid_t grp_user;
    grp_user = H5Gcreate2(file, "user", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
////////////////////////////////  group "user"  attribute "name"   
    
    if(strlen(params->USER.user_name) != 0)
    {
    writeAttributeString(grp_user, "name",  params->USER.user_name);
    }
    
////////////////////////////////  group "user"  attribute "institution" 
    
    if(strlen(params->USER.institution) != 0)
    {
    writeAttributeString(grp_user, "institution",  params->USER.institution);
    }
    
////////////////////////////////  group "user"  attribute "department"   
    
    if(strlen(params->USER.department) != 0)
    {
    writeAttributeString(grp_user, "department",  params->USER.department);    
    }
////////////////////////////////  group "user"  attribute "email"  
    
    if(strlen(params->USER.email) != 0)
    {
    writeAttributeString(grp_user, "email",  params->USER.email);    
    }
    H5Gclose(grp_user);
//     
////////////////////////////////  group "comments"  
//      fprintf (stderr, " error comments\n " );
        
    hid_t grp_comments;
    grp_comments = H5Gcreate2(file, "comments", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    if(strlen(params->COMMENT.comments) != 0)
    {
    writeAttributeString(grp_comments, "comment",  params->COMMENT.comments);
    }
    
    H5Gclose(grp_comments);   

    H5Fclose(file);
  
    free(Z_h );
    free(xyzCoord_h );
    free(xCoord_h);
    free(yCoord_h);  
    free(zCoord_h); 
    free(specimen_tilt_x);
    free(specimen_tilt_y);
    free(beam_tilt_x);
    free(beam_tilt_y);
    free( DWF_h );
    free( occ_h );
}


void writeHdf5 (const char* filename,  params_t* params,int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d)
{
	int nAt = params->SAMPLE.nAt;
    	int *Z_h;
	float *xyzCoord_h, *DWF_h, *occ_h;
	float *xCoord_h, *yCoord_h, *zCoord_h;
	Z_h = ( int* ) malloc ( nAt * sizeof ( int ) );
	xyzCoord_h = ( float* ) malloc ( nAt * 3 * sizeof ( float ) );
 	xCoord_h = ( float* ) malloc ( nAt *  sizeof ( float ) );
	yCoord_h = ( float* ) malloc ( nAt *  sizeof ( float ) );
	zCoord_h = ( float* ) malloc ( nAt *  sizeof ( float ) );
	DWF_h = ( float* ) malloc ( nAt * sizeof ( float ) );
	occ_h = ( float* ) malloc ( nAt * sizeof ( float ) );

	cuda_assert ( cudaMemcpy ( Z_h, *Z_d, nAt * sizeof ( int ), cudaMemcpyDeviceToHost) );
	cuda_assert ( cudaMemcpy ( xyzCoord_h, *xyzCoord_d, nAt * 3 * sizeof ( int ), cudaMemcpyDeviceToHost ) );
	cuda_assert ( cudaMemcpy ( DWF_h, *DWF_d, nAt * sizeof ( float ), cudaMemcpyDeviceToHost ) );
	cuda_assert ( cudaMemcpy ( occ_h, *occ_d, nAt * sizeof ( float ), cudaMemcpyDeviceToHost ) );

        for (int i =0;i < nAt; i++)
	{
	  xCoord_h[i]= xyzCoord_h[i*3+0];
	  yCoord_h[i]= xyzCoord_h[i*3+1];
	  zCoord_h[i]= xyzCoord_h[i*3+2];
	}
  
 
      hid_t file;         /* file and dataset handles */
      file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); 
/////////////////////////// attribute "version"

      float version_value= version;
      writeAttributeSingle(file,"version",version_value);  
    
     
/////////////////////////// group  "data"
    hid_t grp_data;
    grp_data = H5Gcreate2(file, "/data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    


   if (printLevel>0 && params->IM.m1*params->IM.m2*params->IM.m3<268435456)
   {

    ///////////////////////// subgroup of "data" potential_slices          
    hid_t grp_potential_slice;
    grp_potential_slice = H5Gcreate2(grp_data, "potential_slices", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
   /////////////////////// group "potential_slices" attribute "emd_group_type"    
   
    unsigned char emd_group_type_value = 1;
    writeAttributeSingle(grp_potential_slice,"emd_group_type",emd_group_type_value);  
    
///////////////////////////// group "potential_slices" dataset "data"
   

   


   float * p_xyz;


   p_xyz = ( float* ) malloc ( params->IM.m1* params->IM.m2*  params->IM.m3 * 2*  sizeof ( float ) );
   
   for (int i= 0; i <params->IM.m3; i++)
      for (int j= 0; j <params->IM.m2; j++)
       for (int k= 0; k <params->IM.m1; k++)
	  {
	 p_xyz[2*(k*params->IM.m3 *params->IM.m2 + j *params->IM.m3 + i) + 0 ] = 0;
         p_xyz[2*(k*params->IM.m3 *params->IM.m2 + j *params->IM.m3 + i) + 1 ] = 0;
	  }
     
   hid_t  dsp_potential_data;
   dsp_potential_data = H5Screate(H5S_SIMPLE);
   int data_potential_RANK = 4;
   hsize_t data_potential_dim[] = {(long unsigned int)params->IM.m1, (long unsigned int)params->IM.m2,  (long unsigned int)params->IM.m3, 2};
   H5Sset_extent_simple(dsp_potential_data, data_potential_RANK, data_potential_dim, NULL);
   hid_t dst_potential_data;
   dst_potential_data = H5Dcreate2(grp_potential_slice, "data", H5T_NATIVE_FLOAT, dsp_potential_data, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_potential_data, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, p_xyz);
   H5Sclose(dsp_potential_data); 
   H5Dclose(dst_potential_data);
// //    
   /////////////////////// group "potential_slices" dataset "dim1" 
   
   float * potential_x_coordinates;
   potential_x_coordinates = ( float* ) malloc ( params->IM.m1 *  sizeof ( float ) );
   for (int i=0; i<params->IM.m1; i++)
   {
     potential_x_coordinates[i] = i -(params->IM.m1-1)/2.0;
   }
   
   hid_t  dsp_potential_dim1;
   dsp_potential_dim1 = H5Screate(H5S_SIMPLE);
   int data_potential_dim1_RANK = 1;
   hsize_t data_potential_dim1_dim[] = {(long unsigned int)params->IM.m1};
   H5Sset_extent_simple(dsp_potential_dim1, data_potential_dim1_RANK, data_potential_dim1_dim, NULL);
   hid_t dst_potential_dim1;
   dst_potential_dim1 = H5Dcreate2(grp_potential_slice, "dim1", H5T_NATIVE_FLOAT, dsp_potential_dim1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_potential_dim1, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, potential_x_coordinates);
   free(potential_x_coordinates);
  
    
//////////////////////////////  group "potential_slices" dataset "dim1" attribute "name"

   writeAttributeString(dst_potential_dim1, "name",  "x");

////////////////////////////// group "potential_slices" dataset "dim1" attribute "units"   

   writeAttributeString(dst_potential_dim1, "units",  "[m]");
   
   H5Sclose(dsp_potential_dim1); 
   H5Dclose(dst_potential_dim1); 
   
//// ///////////////////////// group "potential_slices" dataset "dim2" 
   float * potential_y_coordinates;
   potential_y_coordinates = ( float* ) malloc ( params->IM.m2 *  sizeof ( float ) );
   for (int i=0; i<params->IM.m2; i++)
   {
     potential_y_coordinates[i] = i -(params->IM.m2-1)/2.0;
   }

   
   hid_t  dsp_potential_dim2;
   dsp_potential_dim2 = H5Screate(H5S_SIMPLE);
   int data_potential_dim2_RANK = 1;
   hsize_t data_potential_dim2_dim[] = {(long unsigned int)params->IM.m2};
   H5Sset_extent_simple(dsp_potential_dim2, data_potential_dim2_RANK, data_potential_dim2_dim, NULL);
   hid_t dst_potential_dim2;
   dst_potential_dim2 = H5Dcreate2(grp_potential_slice, "dim2", H5T_NATIVE_FLOAT, dsp_potential_dim2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_potential_dim2, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, potential_y_coordinates);
   free(potential_y_coordinates);
   

   
////////////////////////////// group "potential_slices" dataset "dim2" attribute "name"
   writeAttributeString(dst_potential_dim2, "name",  "y");

   
////////////////////////////// group "potential_slices" dataset "dim2" attribute "units"   
 
   writeAttributeString(dst_potential_dim2, "units",  "[m]");
   
   H5Sclose(dsp_potential_dim2); 
   H5Dclose(dst_potential_dim2); 
    
/////////////////////////////// group "potential_slices" dataset "dim3" 
    
   float * potential_z_coordinates;
   potential_z_coordinates = ( float* ) malloc ( params->IM.m3 *  sizeof ( float ) );
   for (int i=0; i<params->IM.m3; i++)
   {
     potential_z_coordinates[i] = i -(params->IM.m3-1)/2.0;
   }
   
   hid_t  dsp_potential_dim3;
   dsp_potential_dim3 = H5Screate(H5S_SIMPLE);
   int data_potential_dim3_RANK = 1;
   hsize_t data_potential_dim3_dim[] = {(long unsigned int)params->IM.m3};
   H5Sset_extent_simple(dsp_potential_dim3, data_potential_dim3_RANK, data_potential_dim3_dim, NULL);
   hid_t dst_potential_dim3;
   dst_potential_dim3 = H5Dcreate2(grp_potential_slice, "dim3", H5T_NATIVE_FLOAT, dsp_potential_dim3, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_potential_dim3, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, potential_z_coordinates);
   free(potential_z_coordinates);
  
////////////////////////////// group "potential_slices" dataset "dim3" attribute "name"
   writeAttributeString(dst_potential_dim3, "name",  "z");
// 
////////////////////////////// group "potential_slices" dataset "dim3" attribute "units"   

   writeAttributeString(dst_potential_dim3, "units",  "[m]");
   H5Sclose(dsp_potential_dim3); 
   H5Dclose(dst_potential_dim3); 
   
///////////////////////// group "potential_slices" dataset "dim4" 
    hid_t  dsp_potential_dim4;
    dsp_potential_dim4 = H5Screate(H5S_SIMPLE);
    int data_potential_dim4_RANK = 1;
    hsize_t data_potential_dim4_dim[] = {2};
    const char*  complex_name[2] = {"real", "imag"};
   

   hid_t dim4_data_type;
   int slen = strlen (complex_name[0]);
   fprintf (stderr, " slen= %i \n ",slen );
   
   dim4_data_type = H5Tcopy(H5T_C_S1);
   H5Tset_size(dim4_data_type, slen);
   H5Tset_strpad(dim4_data_type,H5T_STR_NULLTERM);
   H5Sset_extent_simple(dsp_potential_dim4, data_potential_dim4_RANK, data_potential_dim4_dim, NULL);
   hid_t dst_potential_dim4;
   dst_potential_dim4 = H5Dcreate2(grp_potential_slice, "dim4", dim4_data_type, dsp_potential_dim4, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_potential_dim4, dim4_data_type, H5S_ALL , H5S_ALL, H5P_DEFAULT, *complex_name);

// // //    
// // //    
////////////////////////////// group "potential_slices" dataset "dim4" attribute "name"
   hid_t dsp_potential_dim4_name;
   dsp_potential_dim4_name  = H5Screate(H5S_SCALAR);
   char strdim4[] = "complex";                /* Value of the string attribute */
   slen =strlen (strdim4);
   hid_t attrtype;
   hid_t attr_potential_dim4_name;
   attrtype = H5Tcopy(H5T_C_S1);
   H5Tset_size(attrtype, slen);
   H5Tset_strpad(attrtype,H5T_STR_NULLTERM);
   attr_potential_dim4_name = H5Acreate2(dst_potential_dim4, "name", attrtype, dsp_potential_dim4_name, H5P_DEFAULT, H5P_DEFAULT);
   H5Awrite(attr_potential_dim4_name, attrtype, strdim4);
   H5Sclose(dsp_potential_dim4_name);
   H5Aclose(attr_potential_dim4_name);

//////////////////////////////  group "potential_slices" dataset "dim4" attribute "units"   
   writeAttributeString(dst_potential_dim4, "units",  "[]");

   H5Sclose(dsp_potential_dim4); 
   H5Dclose(dst_potential_dim4); 
      
   H5Gclose(grp_potential_slice);
   }
       
// //////////////////////////////  subgroup of "data" images
   hid_t grp_images;
   grp_images = H5Gcreate2(grp_data, "images", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   
/////////////////////////////// group "images" attribute "emd_group_type"    
   unsigned char emd_images_type_value = 1;
   writeAttributeSingle(grp_images,"emd_group_type",emd_images_type_value);  
//    
/////////////////////////////// group "images" dataset "data"
   float * f_xyz;
   f_xyz = ( float* ) malloc ( params->IM.n1 *params->IM.n2 *params->IM.n3 *  sizeof ( float ) );
   
   for (int i= 0; i <params->IM.n3; i++)
      for (int j= 0; j <params->IM.n2; j++)
       for (int k= 0; k <params->IM.n1; k++)
       {
	 f_xyz[k*params->IM.n3 *params->IM.n2 + j *params->IM.n3 + i ] = 0.0;
       }
   
   hid_t  dsp_images_data;
   dsp_images_data = H5Screate(H5S_SIMPLE);
   int data_images_RANK = 3;
   hsize_t data_images_dim[] = {(long unsigned int)params->IM.n1, (long unsigned int)params->IM.n2,(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_images_data, data_images_RANK, data_images_dim, NULL);
   hid_t dst_images_data;
   dst_images_data = H5Dcreate2(grp_images, "data", H5T_NATIVE_FLOAT, dsp_images_data, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_images_data, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, f_xyz);
   H5Sclose(dsp_images_data); 
   H5Dclose(dst_images_data);
   
      ///////////////////////// group "images" dataset "dim1" 
   float * images_x_coordinates;
   images_x_coordinates = ( float* ) malloc ( params->IM.n1 *  sizeof ( float ) );
   for (int i=0; i<params->IM.n1; i++)
   {
     images_x_coordinates[i] = i -(params->IM.n1-1)/2.0;
   }
   
   hid_t  dsp_images_dim1;
   dsp_images_dim1 = H5Screate(H5S_SIMPLE);
   int data_images_dim1_RANK = 1;
   hsize_t data_images_dim1_dim[] = {(long unsigned int)params->IM.n1};
   H5Sset_extent_simple(dsp_images_dim1, data_images_dim1_RANK, data_images_dim1_dim, NULL);
   hid_t dst_images_dim1;
   dst_images_dim1 = H5Dcreate2(grp_images, "dim1", H5T_NATIVE_FLOAT, dsp_images_dim1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_images_dim1, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, images_x_coordinates);
   free(images_x_coordinates);
    
////////////////////////////// group "images" dataset "dim1" attribute "name"
////////////////////////////// group "images" dataset "dim2" attribute "name"
   writeAttributeString(dst_images_dim1, "name",  "x");
// 
//////////////////////////////  group "images" dataset "dim2" attribute "units"   
   writeAttributeString(dst_images_dim1, "units",  "[m]");
   
   H5Sclose(dsp_images_dim1); 
   H5Dclose(dst_images_dim1); 
// //    
   ///////////////////////// group "images" dataset "dim2" 
   
   float * images_y_coordinates;
   images_y_coordinates = ( float* ) malloc ( params->IM.n2 *  sizeof ( float ) );
   for (int i=0; i<params->IM.n2; i++)
   {
     images_y_coordinates[i] = i -(params->IM.n2-1)/2.0;
   }
   
   
   hid_t  dsp_images_dim2;
   dsp_images_dim2 = H5Screate(H5S_SIMPLE);
   int data_images_dim2_RANK = 1;
   hsize_t data_images_dim2_dim[] = {(long unsigned int)params->IM.n2};
   H5Sset_extent_simple(dsp_images_dim2, data_images_dim2_RANK, data_images_dim2_dim, NULL);
   hid_t dst_images_dim2;
   dst_images_dim2 = H5Dcreate2(grp_images, "dim2", H5T_NATIVE_FLOAT, dsp_images_dim2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_images_dim2, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, images_y_coordinates);
   free(images_y_coordinates);
   
////////////////////////////// group "images" dataset "dim2" attribute "name"
   writeAttributeString(dst_images_dim2, "name",  "y");
// 
//////////////////////////////  group "images" dataset "dim2" attribute "units"   
   writeAttributeString(dst_images_dim2, "units",  "[m]");
   
   H5Sclose(dsp_images_dim2); 
   H5Dclose(dst_images_dim2); 
   
   ///////////////////////// group "images" dataset "dim3" 

   float * images_z_index;
   images_z_index = ( float* ) malloc ( params->IM.n3 *  sizeof ( float ) );
   for (int i=0; i<params->IM.n3; i++)
   {
     images_z_index[i] = i;
   }
   
   hid_t  dsp_images_dim3;
   dsp_images_dim3 = H5Screate(H5S_SIMPLE);
   int data_images_dim3_RANK = 1;
   hsize_t data_images_dim3_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_images_dim3, data_images_dim3_RANK, data_images_dim3_dim, NULL);
   hid_t dst_images_dim3;
   dst_images_dim3 = H5Dcreate2(grp_images, "dim3", H5T_NATIVE_FLOAT, dsp_images_dim3, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_images_dim3, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, images_z_index);
   free(images_z_index);
  
////////////////////////////// group "images" dataset "dim3" attribute "name"
   writeAttributeString(dst_images_dim3, "name",  "z");
// 
////////////////////////////// group "images" dataset "dim3" attribute "units"   
   
   writeAttributeString(dst_images_dim3, "units",  "[m]");
//    
   H5Sclose(dsp_images_dim3); 
   H5Dclose(dst_images_dim3); 
    
   H5Gclose(grp_images);
   H5Gclose(grp_data);
 
//     
// fprintf (stderr, " error microscope\n " );
////////////////////////////////  group "microscope" 
   hid_t grp_microscope;
   grp_microscope = H5Gcreate2(file, "/microscope", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

////////////////////////////////  group "microscope" attribute "voltage"
   writeAttributeSingle(grp_microscope,"voltage",params->EM.E0);  
   
////////////////////////////////  group "microscope" attribute "voltage_units"   
   writeAttributeString(grp_microscope, "voltage_units",  "[v]");
   
////////////////////////////////  group "microscope" attribute "gamma"
   writeAttributeSingle(grp_microscope,"gamma",params->EM.gamma);  
   
////////////////////////////////  group "microscope" attribute "wavelength"
   writeAttributeSingle(grp_microscope,"wavelength",params->EM.lambda);  
   
////////////////////////////////  group "microscope" attribute "wavelength_units"
   writeAttributeString(grp_microscope, "wavelength_units",  "[m]");

////////////////////////////////  group "microscope" attribute "interaction_constant"

   writeAttributeSingle(grp_microscope,"interaction_constant",params->EM.sigma);   

////////////////////////////////  group "microscope" attribute "interaction_constant_units"
   writeAttributeString(grp_microscope, "interaction_constant_units",  "[V^-1][m^-1]");
   
////////////////////////////////  group "microscope" attribute "focus_spread"
   writeAttributeSingle(grp_microscope,"focus_spread",params->EM.defocspread);
   
////////////////////////////////  group "microscope" attribute "focus_spread_units"
   writeAttributeString(grp_microscope, "focus_spread_units",  "[m]");
   
////////////////////////////////  group "microscope" attribute "illumination_angle"
   writeAttributeSingle(grp_microscope,"illumination_angle",params->EM.illangle);

////////////////////////////////  group "microscope" attribute "illumination_angle_units"
   writeAttributeString(grp_microscope, "illumination_angle_units",  "[rad]");
   
////////////////////////////////  group "microscope" attribute "objective_aperture"
   writeAttributeSingle(grp_microscope,"objective_aperture",params->EM.ObjAp);
   
////////////////////////////////  group "microscope" attribute "objective_aperture_units"
   writeAttributeString(grp_microscope, "objective_aperture_units",  "[rad]");
   
////////////////////////////////  group "microscope" attribute "mtf_a"
   writeAttributeSingle(grp_microscope,"mtf_a",params->EM.mtfa);

////////////////////////////////  group "microscope" attribute "mtf_b"
   writeAttributeSingle(grp_microscope,"mtf_b",params->EM.mtfb);

////////////////////////////////  group "microscope" attribute "mtf_c"
   writeAttributeSingle(grp_microscope,"mtf_c",params->EM.mtfc);

////////////////////////////////  group "microscope" attribute "mtf_d"   
   writeAttributeSingle(grp_microscope,"mtf_d",params->EM.mtfd);
//    
////////////////////////////////  group "microscope" group "aberrations"   
   hid_t grp_aberrations;
   
   grp_aberrations = H5Gcreate2(grp_microscope, "aberrations", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "amplitude_units"    
   writeAttributeString(grp_aberrations, "amplitude_units",  "[m]");
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "amplitude_units" 
   writeAttributeString(grp_aberrations, "angle_units",  "[rad]");

////////////////////////////////  group "microscope" group "aberrations"  attribute "C1_amplitude" 
   writeAttributeSingle(grp_aberrations,"C1_amplitude",params->EM.aberration.C1_0);

////////////////////////////////  group "microscope" group "aberrations"  attribute "C1_angle" 
//    writeAttributeSingle(grp_aberrations,"C1_angle",params->EM.aberration.C1_1);

////////////////////////////////  group "microscope" group "aberrations"  attribute "A1_amplitude" 
   writeAttributeSingle(grp_aberrations,"A1_amplitude",params->EM.aberration.A1_0);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A1_angle" 
   writeAttributeSingle(grp_aberrations,"A1_angle",params->EM.aberration.A1_1);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A2_amplitude" 
   writeAttributeSingle(grp_aberrations,"A2_amplitude",params->EM.aberration.A2_0);

////////////////////////////////  group "microscope" group "aberrations"  attribute "A2_angle" 
   writeAttributeSingle(grp_aberrations,"A2_angle",params->EM.aberration.A2_1);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "B2_amplitude" 
   writeAttributeSingle(grp_aberrations,"B2_amplitude",params->EM.aberration.B2_0);

////////////////////////////////  group "microscope" group "aberrations"  attribute "B2_angle" 
   writeAttributeSingle(grp_aberrations,"B2_angle",params->EM.aberration.B2_1);   

////////////////////////////////  group "microscope" group "aberrations"  attribute "C3_amplitude" 
   writeAttributeSingle(grp_aberrations,"C3_amplitude",params->EM.aberration.C3_0);  

////////////////////////////////  group "microscope" group "aberrations"  attribute "A3_angle" 
//    writeAttributeSingle(grp_aberrations,"C3_angle",params->EM.aberration.C3_1); 

////////////////////////////////  group "microscope" group "aberrations"  attribute "A3_amplitude" 
   writeAttributeSingle(grp_aberrations,"A3_amplitude",params->EM.aberration.A3_0);     
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A3_angle" 
   writeAttributeSingle(grp_aberrations,"A3_angle",params->EM.aberration.A3_1);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S3_amplitude" 
   writeAttributeSingle(grp_aberrations,"S3_amplitude",params->EM.aberration.S3_0);        
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S3_angle" 
   writeAttributeSingle(grp_aberrations,"S3_angle",params->EM.aberration.S3_1);   
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A4_amplitude" 
   writeAttributeSingle(grp_aberrations,"A4_amplitude",params->EM.aberration.A4_0); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A4_angle" 
   writeAttributeSingle(grp_aberrations,"A4_angle",params->EM.aberration.A4_1);

////////////////////////////////  group "microscope" group "aberrations"  attribute "B4_amplitude" 
   writeAttributeSingle(grp_aberrations,"B4_amplitude",params->EM.aberration.B4_0); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "B4_angle" 
   writeAttributeSingle(grp_aberrations,"B4_angle",params->EM.aberration.B4_1);
 
////////////////////////////////  group "microscope" group "aberrations"  attribute "D4_amplitude" 
   writeAttributeSingle(grp_aberrations,"D4_amplitude",params->EM.aberration.D4_0);   
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "D4_angle" 
   writeAttributeSingle(grp_aberrations,"D4_angle",params->EM.aberration.D4_1); 

////////////////////////////////  group "microscope" group "aberrations"  attribute "C5_amplitude" 
   writeAttributeSingle(grp_aberrations,"C5_amplitude",params->EM.aberration.C5_0);  
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "C5_angle" 
//    writeAttributeSingle(grp_aberrations,"C5_angle",params->EM.aberration.C5_1);   

////////////////////////////////  group "microscope" group "aberrations"  attribute "A5_amplitude" 
   writeAttributeSingle(grp_aberrations,"A5_amplitude",params->EM.aberration.A5_0);      

////////////////////////////////  group "microscope" group "aberrations"  attribute "A5_angle" 
   writeAttributeSingle(grp_aberrations,"A5_angle",params->EM.aberration.A5_1); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "R5_amplitude" 
   writeAttributeSingle(grp_aberrations,"R5_amplitude",params->EM.aberration.R5_0);  
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "R5_angle" 
   writeAttributeSingle(grp_aberrations,"R5_angle",params->EM.aberration.R5_1); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S5_amplitude" 
   writeAttributeSingle(grp_aberrations,"S5_amplitude",params->EM.aberration.S5_0);  
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S5_angle" 
   writeAttributeSingle(grp_aberrations,"S5_angle",params->EM.aberration.S5_1);    
   
   H5Gclose(grp_aberrations);
   H5Gclose(grp_microscope);
 
 
////////////////////////////////  group "sample"     
   hid_t grp_sample;
   grp_sample = H5Gcreate2(file, "/sample", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

////////////////////////////////  group "sample" attribute "name"      
  
   writeAttributeString(grp_sample, "name",  params->SAMPLE.sample_name);

////////////////////////////////  group "sample" attribute "material"      

   writeAttributeString(grp_sample, "material",  params->SAMPLE.material);   
 
// ///////////////////////////////// group "sample" dataset "atomic_numbers" 
   hid_t  dsp_sample_atomic_numbers;
   dsp_sample_atomic_numbers = H5Screate(H5S_SIMPLE);
   int data_sample_atomic_numbers_RANK = 1;
   hsize_t data_sample_atomic_numbers_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_atomic_numbers, data_sample_atomic_numbers_RANK, data_sample_atomic_numbers_dim, NULL);
   hid_t dst_sample_atomic_numbers;
   dst_sample_atomic_numbers = H5Dcreate2(grp_sample, "atomic_numbers", H5T_NATIVE_INT, dsp_sample_atomic_numbers, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_atomic_numbers, H5T_NATIVE_INT, H5S_ALL , H5S_ALL, H5P_DEFAULT, Z_h);   
   H5Dclose(dst_sample_atomic_numbers);
   H5Sclose(dsp_sample_atomic_numbers); 
   
///////////////////////////////// group "sample" dataset "x_coordinates"
   hid_t  dsp_sample_x_coordinates;
   dsp_sample_x_coordinates = H5Screate(H5S_SIMPLE);
   int data_sample_x_coordinates_RANK = 1;
   hsize_t data_sample_x_coordinates_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_x_coordinates, data_sample_x_coordinates_RANK, data_sample_x_coordinates_dim, NULL);
   hid_t dst_sample_x_coordinates;
   dst_sample_x_coordinates = H5Dcreate2(grp_sample, "x_coordinates", H5T_NATIVE_FLOAT, dsp_sample_x_coordinates, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_x_coordinates, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, xCoord_h);   

///////////////////////////////// group "sample" dataset "x_coordinates" attribute "units"   
   writeAttributeString(dst_sample_x_coordinates, "units",  "[m]");
   
   H5Sclose(dsp_sample_x_coordinates);
   H5Dclose(dst_sample_x_coordinates);   
//    
///////////////////////////////// group "sample" dataset "y_coordinates"
   hid_t  dsp_sample_y_coordinates;
   dsp_sample_y_coordinates = H5Screate(H5S_SIMPLE);
   int data_sample_y_coordinates_RANK = 1;
   hsize_t data_sample_y_coordinates_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_y_coordinates, data_sample_y_coordinates_RANK, data_sample_y_coordinates_dim, NULL);
   hid_t dst_sample_y_coordinates;
   dst_sample_y_coordinates = H5Dcreate2(grp_sample, "y_coordinates", H5T_NATIVE_FLOAT, dsp_sample_y_coordinates, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_y_coordinates, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, yCoord_h);   

///////////////////////////////// group "sample" dataset "y_coordinates" attribute "units"   
   writeAttributeString(dst_sample_y_coordinates, "units",  "[m]");

   H5Sclose(dsp_sample_y_coordinates);
   H5Dclose(dst_sample_y_coordinates);  

 //////////////////////////////// group "sample" dataset "z_coordinates"
   
   hid_t  dsp_sample_z_coordinates;
   dsp_sample_z_coordinates = H5Screate(H5S_SIMPLE);
   int data_sample_z_coordinates_RANK = 1;
   hsize_t data_sample_z_coordinates_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_z_coordinates, data_sample_z_coordinates_RANK, data_sample_z_coordinates_dim, NULL);
   hid_t dst_sample_z_coordinates;
   dst_sample_z_coordinates = H5Dcreate2(grp_sample, "z_coordinates", H5T_NATIVE_FLOAT, dsp_sample_z_coordinates, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_z_coordinates, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, zCoord_h);   

///////////////////////////////// group "sample" dataset "z_coordinates" attribute "units"   
   
   writeAttributeString(dst_sample_z_coordinates, "units",  "[m]");

   H5Sclose(dsp_sample_z_coordinates);
   H5Dclose(dst_sample_z_coordinates);   
  
///////////////////////////////// group "sample" dataset "debeye_waller_factors "
   hid_t  dsp_sample_debeye_waller_factors;
   dsp_sample_debeye_waller_factors = H5Screate(H5S_SIMPLE);
   int data_sample_debeye_waller_factors_RANK = 1;
   hsize_t data_sample_debeye_waller_factors_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_debeye_waller_factors, data_sample_debeye_waller_factors_RANK, data_sample_debeye_waller_factors_dim, NULL);
   hid_t dst_sample_debeye_waller_factors;
   dst_sample_debeye_waller_factors = H5Dcreate2(grp_sample, "debeye_waller_factors", H5T_NATIVE_FLOAT, dsp_sample_debeye_waller_factors, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_debeye_waller_factors, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, DWF_h);   

///////////////////////////////// group "sample" dataset "debeye_waller_factors" attribute "units"   
   
   writeAttributeString(dst_sample_debeye_waller_factors, "units",  "[m^2]");

   H5Sclose(dsp_sample_debeye_waller_factors);
   H5Dclose(dst_sample_debeye_waller_factors);   
   
///////////////////////////////// group "sample" dataset "occupancy"
   hid_t  dsp_sample_occupancy;
   dsp_sample_occupancy = H5Screate(H5S_SIMPLE);
   int data_sample_occupancy_RANK = 1;
   hsize_t data_sample_occupancy_dim[] = {(long unsigned int)nAt};
   H5Sset_extent_simple(dsp_sample_occupancy, data_sample_occupancy_RANK, data_sample_occupancy_dim, NULL);
   hid_t dst_sample_occupancy;
   dst_sample_occupancy = H5Dcreate2(grp_sample, "occupancy", H5T_NATIVE_FLOAT, dsp_sample_occupancy, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_sample_occupancy, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, occ_h);   

   H5Sclose(dsp_sample_occupancy);
   H5Dclose(dst_sample_occupancy);      
    
/////////////////////////////// group "sample" attribute "absorptive_potential_factor"   /// ??
    
   writeAttributeSingle(grp_sample,"absorptive_potential_factor",params->SAMPLE.imPot);    
    
    
   H5Gclose(grp_sample);
   
 
   
////////////////////////////////  group "imaging"         
    hid_t grp_imaging;
    grp_imaging = H5Gcreate2(file, "/imaging", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

///////////////////////////////// group "imaging" attribute "mode"      
 
    writeAttributeSingle(grp_imaging,"mode",params->IM.mode);  
//     
///////////////////////////////// group "imaging" attribute "sample_size_x"  

    writeAttributeSingle(grp_imaging,"sample_size_x",params->IM.m1);        

///////////////////////////////// group "imaging" attribute "sample_size_y"  

    writeAttributeSingle(grp_imaging,"sample_size_y",params->IM.m2);       

///////////////////////////////// group "imaging" attribute "sample_size_z"  

    writeAttributeSingle(grp_imaging,"sample_size_z",params->IM.m3);       

///////////////////////////////// group "imaging" attribute "sample_size_units"  

    writeAttributeString(grp_imaging, "sample_size_units",  "[pix]");   
     
///////////////////////////////// group "imaging" attribute "pixel_size_x"  

    writeAttributeSingle(grp_imaging,"pixel_size_x",params->IM.d1);     
    
///////////////////////////////// group "imaging" attribute "pixel_size_y"  

    writeAttributeSingle(grp_imaging,"pixel_size_y",params->IM.d2);       
    
///////////////////////////////// group "imaging" attribute "pixel_size_z"  

    writeAttributeSingle(grp_imaging,"pixel_size_z",params->IM.d3);        
    
///////////////////////////////// group "imaging" attribute "sample_size_units"  

    writeAttributeString(grp_imaging, "pixel_size_units",  "[m]");     
     
///////////////////////////////// group "imaging" attribute "image_size_x"  
    writeAttributeSingle(grp_imaging,"image_size_x",params->IM.n1);     
    
///////////////////////////////// group "imaging" attribute "image_size_y"  
    writeAttributeSingle(grp_imaging,"image_size_y",params->IM.n2);       
    
///////////////////////////////// group "imaging" attribute "image_size_z"  
    writeAttributeSingle(grp_imaging,"image_size_z",params->IM.n3);    
    
///////////////////////////////// group "imaging" attribute "image_size_units"  
    writeAttributeString(grp_imaging, "image_size_units",  "[pix]");      
//     
///////////////////////////////// group "imaging" attribute "border_size_x"  
    writeAttributeSingle(grp_imaging,"border_size_x",params->IM.dn1);     
    
///////////////////////////////// group "imaging" attribute "border_size_y"  
    writeAttributeSingle(grp_imaging,"border_size_y",params->IM.dn2);   
    
///////////////////////////////// group "imaging" attribute "border_size_units"  
    writeAttributeString(grp_imaging, "border_size_units",  "[pix]");      
// //     
///////////////////////////////// group "imaging" attribute "specimen_tilt_offset_x"  
    writeAttributeSingle(grp_imaging,"specimen_tilt_offset_x",params->IM.specimen_tilt_offset_x);     
    
///////////////////////////////// group "imaging" attribute "specimen_tilt_offset_y"  
    writeAttributeSingle(grp_imaging,"specimen_tilt_offset_y",params->IM.specimen_tilt_offset_y);       
    
///////////////////////////////// group "imaging" attribute "specimen_tilt_offset_z"  
    writeAttributeSingle(grp_imaging,"specimen_tilt_offset_z",params->IM.specimen_tilt_offset_z);    
//     
///////////////////////////////// group "imaging" attribute "image_size_units"  
    writeAttributeString(grp_imaging, "specimen_tilt_offset_units",  "[rad]");   
//     
///////////////////////////////// group "imaging" attribute "frozen_phonons"  
    writeAttributeSingle(grp_imaging,"frozen_phonons",params->IM.frPh);     
    
///////////////////////////////// group "imaging" attribute "pixel_dose"  
    writeAttributeSingle(grp_imaging,"pixel_dose",params->IM.pD);       
    
///////////////////////////////// group "imaging" attribute "subpixel_size_z"  
    writeAttributeSingle(grp_imaging,"subpixel_size_z",params->IM.subSlTh);
    
///////////////////////////////// group "imaging" attribute "image_size_units"  
    writeAttributeString(grp_imaging, "subpixel_size_units",  "[m]");   
    
      
    float *specimen_tilt_x, *specimen_tilt_y;
    specimen_tilt_x = ( float* ) malloc (  params->IM.n3*  sizeof ( float ) );
    specimen_tilt_y = ( float* ) malloc (  params->IM.n3*  sizeof ( float ) );
    
    for(int i=0; i< params->IM.n3; i++ )
    {
      specimen_tilt_x[i] = params->IM.tiltspec[2*i+0];
      specimen_tilt_y[i] = params->IM.tiltspec[2*i+1];
    }
    
// ///////////////////////////////// group "imaging" dataset "specimen_tilt_x" 
   hid_t  dsp_imaging_specimen_tilt_x;
   dsp_imaging_specimen_tilt_x = H5Screate(H5S_SIMPLE);
   int data_imaging_specimen_tilt_x_RANK = 1;
   hsize_t data_imaging_specimen_tilt_x_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_imaging_specimen_tilt_x, data_imaging_specimen_tilt_x_RANK, data_imaging_specimen_tilt_x_dim, NULL);
   hid_t dst_imaging_specimen_tilt_x;
   dst_imaging_specimen_tilt_x = H5Dcreate2(grp_imaging, "specimen_tilt_x", H5T_NATIVE_FLOAT, dsp_imaging_specimen_tilt_x, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_imaging_specimen_tilt_x, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, specimen_tilt_x); 
   

   
///////////////////////////////// group "imaging" dataset "specimen_tilt_x"  attribute "units"  
   writeAttributeString(dst_imaging_specimen_tilt_x, "units",  "[rad]");   
    
   H5Dclose(dst_imaging_specimen_tilt_x);
   H5Sclose(dsp_imaging_specimen_tilt_x);    
    
// ///////////////////////////////// group "imaging" dataset "specimen_tilt_y" 
   hid_t  dsp_imaging_specimen_tilt_y;
   dsp_imaging_specimen_tilt_y = H5Screate(H5S_SIMPLE);
   int data_imaging_specimen_tilt_y_RANK = 1;
   hsize_t data_imaging_specimen_tilt_y_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_imaging_specimen_tilt_y, data_imaging_specimen_tilt_y_RANK, data_imaging_specimen_tilt_y_dim, NULL);
   hid_t dst_imaging_specimen_tilt_y;
   dst_imaging_specimen_tilt_y = H5Dcreate2(grp_imaging, "specimen_tilt_y", H5T_NATIVE_FLOAT, dsp_imaging_specimen_tilt_y, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_imaging_specimen_tilt_y, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, specimen_tilt_y); 
   
///////////////////////////////// group "imaging" dataset "specimen_tilt_y"  attribute "units"  
   writeAttributeString(dst_imaging_specimen_tilt_y, "units",  "[rad]");   
    
   H5Dclose(dst_imaging_specimen_tilt_y);
   H5Sclose(dsp_imaging_specimen_tilt_y);   
   
   
    float *beam_tilt_x, *beam_tilt_y;
    beam_tilt_x = ( float* ) malloc (  params->IM.n3*  sizeof ( float ) );
    beam_tilt_y = ( float* ) malloc (  params->IM.n3*  sizeof ( float ) );
    
    for(int i=0; i< params->IM.n3; i++ )
    {
      beam_tilt_x[i] = params->IM.tiltbeam[2*i+0];
      beam_tilt_y[i] = params->IM.tiltbeam[2*i+1];
    }
// ///////////////////////////////// group "imaging" dataset "beam_tilt_x" 
   hid_t  dsp_imaging_beam_tilt_x;
   dsp_imaging_beam_tilt_x = H5Screate(H5S_SIMPLE);
   int data_imaging_beam_tilt_x_RANK = 1;
   hsize_t data_imaging_beam_tilt_x_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_imaging_beam_tilt_x, data_imaging_beam_tilt_x_RANK, data_imaging_beam_tilt_x_dim, NULL);
   hid_t dst_imaging_beam_tilt_x;
   dst_imaging_beam_tilt_x = H5Dcreate2(grp_imaging, "beam_tilt_x", H5T_NATIVE_FLOAT, dsp_imaging_beam_tilt_x, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_imaging_beam_tilt_x, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, beam_tilt_x); 
   
///////////////////////////////// group "imaging" dataset "beam_tilt_x"  attribute "units"  
   writeAttributeString(dst_imaging_beam_tilt_x, "units",  "[rad]");   
//     
   H5Dclose(dst_imaging_beam_tilt_x);
   H5Sclose(dsp_imaging_beam_tilt_x); 
// 
// ///////////////////////////////// group "imaging" dataset "beam_tilt_y" 
   hid_t  dsp_imaging_beam_tilt_y;
   dsp_imaging_beam_tilt_y = H5Screate(H5S_SIMPLE);
   int data_imaging_beam_tilt_y_RANK = 1;
   hsize_t data_imaging_beam_tilt_y_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_imaging_beam_tilt_y, data_imaging_beam_tilt_y_RANK, data_imaging_beam_tilt_y_dim, NULL);
   hid_t dst_imaging_beam_tilt_y;
   dst_imaging_beam_tilt_y = H5Dcreate2(grp_imaging, "beam_tilt_y", H5T_NATIVE_FLOAT, dsp_imaging_beam_tilt_y, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_imaging_beam_tilt_y, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, beam_tilt_y); 
//    
///////////////////////////////// group "imaging" dataset "beam_tilt_y"  attribute "units"  
   writeAttributeString(dst_imaging_beam_tilt_y, "units",  "[rad]");   
    
   H5Dclose(dst_imaging_beam_tilt_y);
   H5Sclose(dsp_imaging_beam_tilt_y); 
// 
///////////////////////////////// group "imaging" dataset "defoci" 
   hid_t  dsp_imaging_defoci;
   dsp_imaging_defoci = H5Screate(H5S_SIMPLE);
   int data_imaging_defoci_RANK = 1;
   hsize_t data_imaging_defoci_dim[] = {(long unsigned int)params->IM.n3};
   H5Sset_extent_simple(dsp_imaging_defoci, data_imaging_defoci_RANK, data_imaging_defoci_dim, NULL);
   hid_t dst_imaging_defoci;
   dst_imaging_defoci = H5Dcreate2(grp_imaging, "defoci", H5T_NATIVE_FLOAT, dsp_imaging_defoci, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dst_imaging_defoci, H5T_NATIVE_FLOAT, H5S_ALL , H5S_ALL, H5P_DEFAULT, params->IM.defoci); 
   
///////////////////////////////// group "imaging" dataset "defoci"  attribute "units"  
   writeAttributeString(dst_imaging_defoci, "units",  "[rad]");   
//     
   H5Dclose(dst_imaging_defoci);
   H5Sclose(dsp_imaging_defoci); 
   H5Gclose(grp_imaging);  

    
////////////////////////////////  group "user"       
    hid_t grp_user;
    grp_user = H5Gcreate2(file, "user", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
////////////////////////////////  group "user"  attribute "name"   
    if(strlen(params->USER.user_name) != 0)
    {
    writeAttributeString(grp_user, "name",  params->USER.user_name);
    }
    
////////////////////////////////  group "user"  attribute "institution" 
    if(strlen(params->USER.institution) != 0)
    {
    writeAttributeString(grp_user, "institution",  params->USER.institution);
    }
////////////////////////////////  group "user"  attribute "department"   
    if(strlen(params->USER.department) != 0)
    {
    writeAttributeString(grp_user, "department",  params->USER.department);    
    }
////////////////////////////////  group "user"  attribute "email"  
    if(strlen(params->USER.email) != 0)
    {
    writeAttributeString(grp_user, "email",  params->USER.email);    
    }
    H5Gclose(grp_user);
//     
////////////////////////////////  group "comments"  
//      fprintf (stderr, " error comments\n " );
        
    hid_t grp_comments;
    grp_comments = H5Gcreate2(file, "comments", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if(strlen(params->COMMENT.comments) != 0)
    {
    writeAttributeString(grp_comments, "comment",  params->COMMENT.comments);
    }
    H5Gclose(grp_comments);   

    H5Fclose(file);
//     
    
    free(Z_h );
    free(xyzCoord_h );
    free(xCoord_h);
    free(yCoord_h);  
    free(zCoord_h); 
    
    free(specimen_tilt_x);
    free(specimen_tilt_y);
    
    free(beam_tilt_x);
    free(beam_tilt_y);
    
    free( DWF_h );
    free( occ_h );
 
    
 
}



bool readHdf5 (const char* filename,  params_t** params,int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d)
{
  
   bool ret =true;
   hid_t  file, dataset, dataspace, dtype;          
   hid_t grp;
   
   hsize_t    * dims_out;
   int rank;
   file  = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
 
   int n3;
///////////////////////////// group "imaging" 
   grp  = H5Gopen2(file, "imaging", H5P_DEFAULT);
   ret =readAttributeSingle(grp,"image_size_z",&n3);
   if(!ret)
   {
     fprintf(stderr, "input emd file missed the specification of image_size_z\n");
     exit(EXIT_FAILURE);
   }
   allocParams ( params, n3 );
   (**params).IM.n3 = n3;
  
   if(!readAttributeSingle(grp,"image_size_x", &(**params).IM.n1))
   {
     fprintf(stderr, "input emd file missed the specification of image_size_x\n");
     exit(EXIT_FAILURE);
   }
   
   if(!readAttributeSingle(grp,"image_size_y", &(**params).IM.n2))
   {
     fprintf(stderr, "input emd file missed the specification of image_size_y\n");
     exit(EXIT_FAILURE);     
   }
   
    if(!readAttributeSingle(grp,"mode",&(**params).IM.mode))
    {
     fprintf(stderr, "input emd file missed the specification of mode\n");
     exit(EXIT_FAILURE);   
    }
    
///////////////////////////////// group "imaging" attribute "sample_size_x"  

    if(!readAttributeSingle(grp,"sample_size_x",&(**params).IM.m1))
    {
     fprintf(stderr, "input emd file missed the specification of sample_size_x\n");
     exit(EXIT_FAILURE);   
    }

///////////////////////////////// group "imaging" attribute "sample_size_y"  
    if(!readAttributeSingle(grp,"sample_size_y",&(**params).IM.m2))
    {     
      fprintf(stderr, "input emd file missed the specification of sample_size_y\n");
      exit(EXIT_FAILURE);   
    }

///////////////////////////////// group "imaging" attribute "sample_size_z"  
    if(!readAttributeSingle(grp,"sample_size_z",&(**params).IM.m3))      
    {     
      fprintf(stderr, "input emd file missed the specification of sample_size_z\n");
      exit(EXIT_FAILURE);   
    }
    
///////////////////////////////// group "imaging" attribute "pixel_size_x"  
    if(!readAttributeSingle(grp,"pixel_size_x",&(**params).IM.d1))    
    {     
      fprintf(stderr, "input emd file missed the specification of pixel_size_x\n");
      exit(EXIT_FAILURE);   
    }
///////////////////////////////// group "imaging" attribute "pixel_size_y"  
    if(!readAttributeSingle(grp,"pixel_size_y",&(**params).IM.d2))      
    {     
      fprintf(stderr, "input emd file missed the specification of pixel_size_y\n");
      exit(EXIT_FAILURE);   
    }    
///////////////////////////////// group "imaging" attribute "pixel_size_z"  
    if(!readAttributeSingle(grp,"pixel_size_z",&(**params).IM.d3))       
    {     
      fprintf(stderr, "input emd file missed the specification of pixel_size_z\n");
      exit(EXIT_FAILURE);   
    }    
///////////////////////////////// group "imaging" attribute "border_size_x"  
    if(!readAttributeSingle(grp,"border_size_x",&(**params).IM.dn1))    
    {     
      fprintf(stderr, "input emd file missed the specification of border_size_x\n");
      exit(EXIT_FAILURE);   
    }    
///////////////////////////////// group "imaging" attribute "border_size_y"  
    if(!readAttributeSingle(grp,"border_size_y",&(**params).IM.dn2))   
    {     
      fprintf(stderr, "input emd file missed the specification of border_size_y\n");
      exit(EXIT_FAILURE);   
    }    
   
///////////////////////////////// group "imaging" attribute "specimen_tilt_offset_x"  

    readAttributeSingle(grp,"specimen_tilt_offset_x",&(**params).IM.specimen_tilt_offset_x);     
  
///////////////////////////////// group "imaging" attribute "specimen_tilt_offset_y"  

    readAttributeSingle(grp,"specimen_tilt_offset_y",&(**params).IM.specimen_tilt_offset_y);       
    
///////////////////////////////// group "imaging" attribute "specimen_tilt_offset_z"  

    readAttributeSingle(grp,"specimen_tilt_offset_z",&(**params).IM.specimen_tilt_offset_z);    
    
///////////////////////////////// group "imaging" attribute "frozen_phonons"  
   
    readAttributeSingle(grp,"frozen_phonons",&(**params).IM.frPh);     
    
///////////////////////////////// group "imaging" attribute "pixel_dose"  
    
    readAttributeSingle(grp,"pixel_dose",&(**params).IM.pD);       
    
///////////////////////////////// group "imaging" attribute "subpixel_size_z"  
    
    readAttributeSingle(grp,"subpixel_size_z",&(**params).IM.subSlTh);
    
///////////////////////////////// group "imaging" dataset 

    float *specimen_tilt_x, *specimen_tilt_y;
    specimen_tilt_x = ( float* ) malloc ((**params).IM.n3*  sizeof ( float ) );
    specimen_tilt_y = ( float* ) malloc ((**params).IM.n3*  sizeof ( float ) );
    
////////////////////////////////  group "imaging" dataset "specimen_tilt_x"         
    dataset = H5Dopen2(grp, "specimen_tilt_x", H5P_DEFAULT);
   
    dataspace = H5Dget_space(dataset);    /* dataspace handle */
    rank      = H5Sget_simple_extent_ndims(dataspace);
    dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
   
    H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
   
    if(dims_out[0]!=(unsigned int)(**params).IM.n3)
    {
          fprintf (stderr, "Reading error \n ");
      return false;
    }
      
      
   dtype =  H5Dget_type(dataset); 
   H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, specimen_tilt_x); 
   H5Sclose(dataspace);
   H5Dclose(dataset);  
   
  ////////////////////////////////  group "imaging" dataset "specimen_tilt_y"         
   dataset = H5Dopen2(grp, "specimen_tilt_y", H5P_DEFAULT);
   
   dataspace = H5Dget_space(dataset);    /* dataspace handle */
   rank      = H5Sget_simple_extent_ndims(dataspace);
   dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
   
   H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
   
   if(dims_out[0]!= (unsigned int)(**params).IM.n3)
   {
          fprintf (stderr, "Reading error!! \n ");
     return false;
   }
      
      
   dtype =  H5Dget_type(dataset); 
   H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, specimen_tilt_y); 
   H5Sclose(dataspace);
   H5Dclose(dataset);  
    

    
    for(int i=0; i< (**params).IM.n3; i++ )
    {
      (**params).IM.tiltspec[2*i+0] = specimen_tilt_x[i]; 
      (**params).IM.tiltspec[2*i+1] = specimen_tilt_y[i];
    }
        
/////////////////////////////////////////////////////////////////////////    
    float *beam_tilt_x, *beam_tilt_y;
    beam_tilt_x = ( float* ) malloc (  (**params).IM.n3*  sizeof ( float ) );
    beam_tilt_y = ( float* ) malloc (  (**params).IM.n3*  sizeof ( float ) );

////////////////////////////////  group "imaging" dataset "beam_tilt_x"         
   dataset = H5Dopen2(grp, "beam_tilt_x", H5P_DEFAULT);
   
   dataspace = H5Dget_space(dataset);    /* dataspace handle */
   rank      = H5Sget_simple_extent_ndims(dataspace);
   dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
   
   H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
   
   if(dims_out[0]!=(unsigned int)(**params).IM.n3)
   {
          fprintf (stderr, "Reading error!! \n ");
     return false;
   }
      
      
   dtype =  H5Dget_type(dataset); 
   H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, beam_tilt_x); 
   H5Sclose(dataspace);
   H5Dclose(dataset);     
   
////////////////////////////////  group "imaging" dataset "beam_tilt_y"         
    dataset = H5Dopen2(grp, "beam_tilt_y", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);    /* dataspace handle */
    rank      = H5Sget_simple_extent_ndims(dataspace);
    dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
    H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
   
    if(dims_out[0]!=(unsigned int)(**params).IM.n3)
   {
     fprintf (stderr, "Reading error!! \n ");
     return false;
   }
         
    dtype =  H5Dget_type(dataset); 
    H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, beam_tilt_y); 
    H5Sclose(dataspace);
    H5Dclose(dataset);    
   
///////////////////////////////////////////////////////////////////////   
    for(int i=0; i< (**params).IM.n3; i++ )
    {
      (**params).IM.tiltbeam[2*i+0] = beam_tilt_x[i];
      (**params).IM.tiltbeam[2*i+1] = beam_tilt_y[i];
    }

////////////////////////////////  group "imaging" dataset "defoci"         

    dataset = H5Dopen2(grp, "defoci", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);    /* dataspace handle */
    rank      = H5Sget_simple_extent_ndims(dataspace);
    dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
    H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
   
    if(dims_out[0]!=(unsigned int)(**params).IM.n3)
    {
     fprintf (stderr, "Reading error!! \n ");
     return false;
    }    
      
    dtype =  H5Dget_type(dataset); 
    H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, (**params).IM.defoci); 
    H5Sclose(dataspace);
    H5Dclose(dataset);        
    H5Gclose(grp);
   
   
    grp  = H5Gopen2(file, "microscope", H5P_DEFAULT);
   
 
////////////////////////////////  group "microscope" attribute "voltage  

    readAttributeSingle(grp,"voltage",&(**params).EM.E0);  
 
////////////////////////////////  group "microscope" attribute "gamma"

    readAttributeSingle(grp,"gamma",&(**params).EM.gamma);  
   
////////////////////////////////  group "microscope" attribute "wavelength"

    readAttributeSingle(grp,"wavelength",&(**params).EM.lambda);  
   
////////////////////////////////  group "microscope" attribute "interaction_constant"

    readAttributeSingle(grp,"interaction_constant",&(**params).EM.sigma);   


////////////////////////////////  group "microscope" attribute "focus_spread"
    readAttributeSingle(grp,"focus_spread",&(**params).EM.defocspread);
   
////////////////////////////////  group "microscope" attribute "illumination_angle"
   
    readAttributeSingle(grp,"illumination_angle",&(**params).EM.illangle);

////////////////////////////////  group "microscope" attribute "objective_aperture"
  
    readAttributeSingle(grp,"objective_aperture",&(**params).EM.ObjAp);
   
////////////////////////////////  group "microscope" attribute "mtf_a"
   
    readAttributeSingle(grp,"mtf_a",&(**params).EM.mtfa);

////////////////////////////////  group "microscope" attribute "mtf_b"
   
    readAttributeSingle(grp,"mtf_b",&(**params).EM.mtfb);

////////////////////////////////  group "microscope" attribute "mtf_c"
   
    readAttributeSingle(grp,"mtf_c",&(**params).EM.mtfc);

////////////////////////////////  group "microscope" attribute "mtf_d"   
   
    readAttributeSingle(grp,"mtf_d",&(**params).EM.mtfd);
     
    H5Gclose(grp);
   
    grp  = H5Gopen2(file, "microscope/aberrations", H5P_DEFAULT);

////////////////////////////////  group "microscope" group "aberrations"  attribute "C1_amplitude" 
    
    readAttributeSingle(grp,"C1_amplitude",&(**params).EM.aberration.C1_0);

////////////////////////////////  group "microscope" group "aberrations"  attribute "A1_amplitude" 
   
    readAttributeSingle(grp,"A1_amplitude",&(**params).EM.aberration.A1_0);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A1_angle" 
  
    readAttributeSingle(grp,"A1_angle",&(**params).EM.aberration.A1_1);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A2_amplitude" 
   
    readAttributeSingle(grp,"A2_amplitude",&(**params).EM.aberration.A2_0);

////////////////////////////////  group "microscope" group "aberrations"  attribute "A2_angle" 
   
    readAttributeSingle(grp,"A2_angle",&(**params).EM.aberration.A2_1);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "B2_amplitude" 
   
    readAttributeSingle(grp,"B2_amplitude",&(**params).EM.aberration.B2_0);

////////////////////////////////  group "microscope" group "aberrations"  attribute "B2_angle" 
  
    readAttributeSingle(grp,"B2_angle",&(**params).EM.aberration.B2_1);   

////////////////////////////////  group "microscope" group "aberrations"  attribute "C3_amplitude" 
   
    readAttributeSingle(grp,"C3_amplitude",&(**params).EM.aberration.C3_0);  

////////////////////////////////  group "microscope" group "aberrations"  attribute "A3_amplitude" 
   
    readAttributeSingle(grp,"A3_amplitude",&(**params).EM.aberration.A3_0);     
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A3_angle" 
   
    readAttributeSingle(grp,"A3_angle",&(**params).EM.aberration.A3_1);
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S3_amplitude" 
   
    readAttributeSingle(grp,"S3_amplitude",&(**params).EM.aberration.S3_0);        
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S3_angle" 
   
    readAttributeSingle(grp,"S3_angle",&(**params).EM.aberration.S3_1);   
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A4_amplitude" 
   
    readAttributeSingle(grp,"A4_amplitude",&(**params).EM.aberration.A4_0); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A4_angle" 
   
    readAttributeSingle(grp,"A4_angle",&(**params).EM.aberration.A4_1);

////////////////////////////////  group "microscope" group "aberrations"  attribute "B4_amplitude" 
   
    readAttributeSingle(grp,"B4_amplitude",&(**params).EM.aberration.B4_0); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "B4_angle" 
   
    readAttributeSingle(grp,"B4_angle",&(**params).EM.aberration.B4_1);
 
////////////////////////////////  group "microscope" group "aberrations"  attribute "D4_amplitude" 
  
    readAttributeSingle(grp,"D4_amplitude",&(**params).EM.aberration.D4_0);   
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "D4_angle" 
   
    readAttributeSingle(grp,"D4_angle",&(**params).EM.aberration.D4_1); 

////////////////////////////////  group "microscope" group "aberrations"  attribute "C5_amplitude" 
   
    readAttributeSingle(grp,"C5_amplitude",&(**params).EM.aberration.C5_0);  
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "A5_amplitude" 
   
    readAttributeSingle(grp,"A5_amplitude",&(**params).EM.aberration.A5_0);      

////////////////////////////////  group "microscope" group "aberrations"  attribute "A5_angle" 
   
    readAttributeSingle(grp,"A5_angle",&(**params).EM.aberration.A5_1); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "R5_amplitude" 
   
    readAttributeSingle(grp,"R5_amplitude",&(**params).EM.aberration.R5_0);  
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "R5_angle" 
   
    readAttributeSingle(grp,"R5_angle",&(**params).EM.aberration.R5_1); 
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S5_amplitude" 
   
    readAttributeSingle(grp,"S5_amplitude",&(**params).EM.aberration.S5_0);  
   
////////////////////////////////  group "microscope" group "aberrations"  attribute "S5_angle" 
   
    readAttributeSingle(grp,"S5_angle",&(**params).EM.aberration.S5_1);   
   
    H5Gclose(grp);
   
   
    grp  = H5Gopen2(file, "user", H5P_DEFAULT);
////////////////////////////////  group "user"  attribute "name"    
    
    readAttributeString(grp, "name",  (**params).USER.user_name);
    
////////////////////////////////  group "user"  attribute "institution"   
    
    readAttributeString(grp, "institution",  (**params).USER.institution);

////////////////////////////////  group "user"  attribute "department"       
    
    readAttributeString(grp, "department",  (**params).USER.department);

////////////////////////////////  group "user"  attribute "email"   
    
    readAttributeString(grp, "email",  (**params).USER.email); 
    H5Gclose(grp);
    
    grp  = H5Gopen2(file, "comments", H5P_DEFAULT);
    readAttributeString(grp, "comment",  (**params).COMMENT.comments);
    H5Gclose(grp);
    grp  = H5Gopen2(file, "sample", H5P_DEFAULT);
    
////////////////////////////////  group "sample" attribute "name"      
   
    readAttributeString(grp, "name",  (**params).SAMPLE.sample_name);

////////////////////////////////  group "sample" attribute "material"      
   
    readAttributeString(grp, "material",  (**params).SAMPLE.material); 
   

   if(!atomsFromExternal)
   {
   
   int *Z_h;
   float *xyzCoord_h, *DWF_h, *occ_h;
   float *xCoord_h, *yCoord_h, *zCoord_h;
   
   ////////////////////////////////  group "sample" dataset "atomic_numbers"       
   dataset = H5Dopen2(grp, "atomic_numbers", H5P_DEFAULT);
   
   dataspace = H5Dget_space(dataset);    /* dataspace handle */
   rank      = H5Sget_simple_extent_ndims(dataspace);
   dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
   
   H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
      
   (**params).SAMPLE.nAt=dims_out[0];
   int nAt = (**params).SAMPLE.nAt;
   
   Z_h = ( int* ) malloc ( nAt * sizeof ( int ) );   
   dtype =  H5Dget_type(dataset); 
   H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, Z_h); 
   H5Sclose(dataspace);
   H5Dclose(dataset);
   
   fprintf (stderr, " Z_h= %i \n ",rank );
   
/////////////////////////////////////////////////////////////////////////////
   
   xyzCoord_h = ( float* ) malloc ( nAt * 3 * sizeof ( float ) );
   DWF_h = ( float* ) malloc ( nAt * sizeof ( float ) );
   occ_h = ( float* ) malloc ( nAt * sizeof ( float ) );
   xCoord_h = ( float* ) malloc ( nAt *  sizeof ( float ) );
   yCoord_h = ( float* ) malloc ( nAt *  sizeof ( float ) );
   zCoord_h = ( float* ) malloc ( nAt *  sizeof ( float ) );
   
   cuda_assert ( cudaMalloc ( ( void** ) Z_d,        nAt* sizeof ( int ) ) );
   cuda_assert ( cudaMalloc ( ( void** ) xyzCoord_d, nAt * 3 * sizeof ( float ) ) );
   cuda_assert ( cudaMalloc ( ( void** ) DWF_d,      nAt * sizeof ( float ) ) );
   cuda_assert ( cudaMalloc ( ( void** ) occ_d,      nAt * sizeof ( float ) ) );
   
   
////////////////////////////////  group "sample" dataset "atomic_numbers"         
   dataset = H5Dopen2(grp, "atomic_numbers", H5P_DEFAULT);
   
   dataspace = H5Dget_space(dataset);    /* dataspace handle */
   rank      = H5Sget_simple_extent_ndims(dataspace);
   dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
   
   H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
   
   if(dims_out[0]!=(unsigned int)nAt)
   {
          fprintf (stderr, "Reading error  \n ");
     return false;
   }
 
   H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, Z_h); 
   H5Sclose(dataspace);
   H5Dclose(dataset);   
   
////////////////////////////////  group "sample" dataset "x_coordinates"         
   dataset = H5Dopen2(grp, "x_coordinates", H5P_DEFAULT);
   
   dataspace = H5Dget_space(dataset);    /* dataspace handle */
   rank      = H5Sget_simple_extent_ndims(dataspace);
   dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
   
   H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
   
   if(dims_out[0]!=(unsigned int)nAt)
   {
          fprintf (stderr, "Reading error \n ");
     return false;
   }
      
      
    dtype =  H5Dget_type(dataset); 
    H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, xCoord_h); 
    H5Sclose(dataspace);
    H5Dclose(dataset);  
   
   
////////////////////////////////  group "sample" dataset "y_coordinates"         
   dataset = H5Dopen2(grp, "y_coordinates", H5P_DEFAULT);
   
   dataspace = H5Dget_space(dataset);    /* dataspace handle */
   rank      = H5Sget_simple_extent_ndims(dataspace);
   dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
   
   H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
   
   if(dims_out[0]!=(unsigned int)nAt)
   {
          fprintf (stderr, "Reading error \n ");
     return false;
   }
      
      
   dtype =  H5Dget_type(dataset); 
   H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, yCoord_h); 
   H5Sclose(dataspace);
   H5Dclose(dataset);  
   
////////////////////////////////  group "sample" dataset "z_coordinates"         
   dataset = H5Dopen2(grp, "z_coordinates", H5P_DEFAULT);
   
   dataspace = H5Dget_space(dataset);    /* dataspace handle */
   rank      = H5Sget_simple_extent_ndims(dataspace);
   dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
   
   H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
   
   if(dims_out[0]!=(unsigned int)nAt)
   {
          fprintf (stderr, "Reading error \n ");
     return false;
   }
      
      
   dtype =  H5Dget_type(dataset); 
   H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, zCoord_h); 
   H5Sclose(dataspace);
   H5Dclose(dataset);  
   
   for (int i =0;i < nAt; i++)
    {
     xyzCoord_h[i*3+0] = xCoord_h[i];
     xyzCoord_h[i*3+1] = yCoord_h[i];
     xyzCoord_h[i*3+2] = zCoord_h[i];
    }
    
    
////////////////////////////////  group "sample" dataset "debeye_waller_factors"         
    dataset = H5Dopen2(grp, "debeye_waller_factors", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);    /* dataspace handle */
    rank      = H5Sget_simple_extent_ndims(dataspace);
    dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
    H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
   
    if(dims_out[0]!=(unsigned int)nAt)
    {
           fprintf (stderr, "Reading error  \n ");
     return false;
    }
      
      
    dtype =  H5Dget_type(dataset); 
    H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, DWF_h); 
    H5Sclose(dataspace);
    H5Dclose(dataset);

////////////////////////////////  group "sample" dataset "occupancy"         
    dataset = H5Dopen2(grp, "occupancy", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);    /* dataspace handle */
    rank      = H5Sget_simple_extent_ndims(dataspace);
    dims_out = (hsize_t * ) malloc (rank  * sizeof ( hsize_t ) );;
   
    H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
   
    if(dims_out[0]!=(unsigned int)nAt)
    {
     fprintf (stderr, "Reading error \n ");
     return false;
    }
     
    dtype =  H5Dget_type(dataset); 
    H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, occ_h); 
    H5Sclose(dataspace);
    H5Dclose(dataset);  
/////////////////////////////////////////////////////////////////////

    cuda_assert ( cudaMemcpy ( *Z_d,       Z_h ,       nAt*sizeof ( int ), cudaMemcpyHostToDevice ) );
    cuda_assert ( cudaMemcpy ( *xyzCoord_d,xyzCoord_h, nAt* 3 * sizeof ( float ), cudaMemcpyHostToDevice ) );
    cuda_assert ( cudaMemcpy ( *DWF_d,     DWF_h,      nAt *sizeof ( float ), cudaMemcpyHostToDevice ) );
    cuda_assert ( cudaMemcpy ( *occ_d,     occ_h,      nAt* sizeof ( float ), cudaMemcpyHostToDevice ) );
    free(Z_h);
    free(xyzCoord_h);  
    free(DWF_h);    
    free(occ_h);
   }
/////////////////////////////////////////////////////////////////////   
    readAttributeSingle(grp,"absorptive_potential_factor",&(**params).SAMPLE.imPot);   
    consitentParams ( *params );
    setGridAndBlockSize ( *params );
    setCufftPlan ( *params );
    setCublasHandle ( *params );
    writeConfig("ParamsUsedEmd.txt",*params, Z_d,  xyzCoord_d, DWF_d, occ_d );
    
    H5Gclose(grp);
    H5Fclose(file);
    return ret;
}


template < typename  Type> void writeAttributeSingle (hid_t loc_id, const char * attrName,  Type value)
{
   hid_t type_id;
   
   if (typeid(int) == typeid(Type))
   {
     type_id = H5T_NATIVE_INT;
   }
   else if(typeid(float) == typeid(Type))
   {
     type_id = H5T_NATIVE_FLOAT;
   }
   else if(typeid(unsigned char) == typeid(Type))
   {
     type_id = H5T_STD_U8LE;
   }
  
 
   hid_t   attr;
   hid_t   ds;
   int     RANK=1;
   hsize_t adim[] = {1};
   
   
   char name[128];
   strcpy (name,attrName);
   ds= H5Screate(H5S_SIMPLE);
   H5Sset_extent_simple(ds, RANK, adim, NULL);
   attr=  H5Acreate2(loc_id, name, type_id, ds, H5P_DEFAULT, H5P_DEFAULT);
   H5Awrite(attr, type_id, &value);
   H5Sclose(ds); 
   H5Aclose(attr);   
}



template < typename  Type> bool readAttributeSingle (hid_t loc_id, const char * attrName,  Type * value)
{
	hid_t type_id;
   
	if (typeid(int) == typeid(Type))
	{
	  type_id = H5T_NATIVE_INT;
	 }
	else if(typeid(float) == typeid(Type))
	{
	 type_id = H5T_NATIVE_FLOAT;
	}
	else if(typeid(unsigned char) == typeid(Type))
	{
	type_id = H5T_STD_U8LE;
	}
  
	hid_t   attr;
	attr = H5Aopen(loc_id, attrName, H5P_DEFAULT);
	
	if(attr<0)
	{
	  return false;
	}
	H5Aread(attr, type_id, value);
	H5Aclose(attr); 
	return true;

}



void writeAttributeString(hid_t loc_id, const char * attrName, const char*  attrString)
{
	hid_t dsp;
	dsp  = H5Screate(H5S_SCALAR);
	int slen =strlen (attrString);
	hid_t attr;
	hid_t attrtype;
	attrtype = H5Tcopy(H5T_C_S1);
	H5Tset_size(attrtype, slen);
	H5Tset_strpad(attrtype,H5T_STR_NULLTERM);
	attr = H5Acreate2(loc_id, attrName, attrtype, dsp, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attr, attrtype, ( const void *)attrString);
	H5Sclose(dsp);
	H5Aclose(attr);   
}


bool readAttributeString(hid_t loc_id, const char * attrName, char*  attrString)
{
	hid_t   attr, atype, atype_mem;
	H5T_class_t  type_class;
	attr = H5Aopen(loc_id, attrName, H5P_DEFAULT);
	if(attr<0)
	{
	  return false;
	}
	
	atype  = H5Aget_type(attr);
	type_class = H5Tget_class(atype);
	if (type_class == H5T_STRING)
	{
           atype_mem = H5Tget_native_type(atype, H5T_DIR_ASCEND);
           H5Aread(attr, atype_mem, attrString);
           H5Tclose(atype_mem);
	 }
           
	H5Aclose(attr);
	H5Tclose(atype);
	return true;
}
