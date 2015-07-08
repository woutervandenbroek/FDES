/*
* modified from http://sourceforge.net/p/gadgetron/gadgetron/ci/development/tree/cmake/FindCUDA/
* 
* This code is licensed under the MIT License. See the FindCUDA.cmake script
* for the text of the license.
*
* Based on code by Christopher Bruns published on Stack Overflow (CC-BY):
* http://stackoverflow.com/questions/2285185
*/
#include <stdio.h>
#include <cuda_runtime.h>



int main() 
{
      int deviceCount, device, sm_major[999], sm_minor[999], compute_major[999], compute_minor[999];
      int gpuDeviceCount = 0;
      struct cudaDeviceProp properties;

      if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
      {
      printf("Couldn't get device count: %s\n", cudaGetErrorString(cudaGetLastError()));
      return 1;
      }
/* machines with no GPUs can still report one emulation device */
      for (device = 0; device < deviceCount; ++device) 
      {
	cudaGetDeviceProperties(&properties, device);
	if (properties.major != 9999) 
	{/* 9999 means emulation only */
	++gpuDeviceCount;}
        sm_major[device] = properties.major;
	sm_minor[device] = properties.minor;
	

	if (sm_major[device] == 2 && sm_minor[device] == 0)
      {
	 compute_major[device]=2; 
	 compute_minor[device]=0;
      }
	
      if (sm_major[device] == 2 && sm_minor[device] == 1)
      {
	 compute_major[device]=2; 
	 compute_minor[device]=0;
      }
      
      if (sm_major[device] == 3 && sm_minor[device] == 0)
      {
	 compute_major[device]=3; 
	 compute_minor[device]=0;
      }
      
      if (sm_major[device] == 3 && sm_minor[device] == 2)
      {
	 compute_major[device]=3; 
	 compute_minor[device]=0;
      }
      
      
       if (sm_major[device] == 3 && sm_minor[device] == 5)
      {
	 compute_major[device]=3; 
	 compute_minor[device]=5;
      }
      
      
      if (sm_major[device] == 5 && sm_minor[device] == 0)
      {
	 compute_major[device]=5; 
	 compute_minor[device]=0;
      }
      
      
      if (sm_major[device] == 5 && sm_minor[device] == 2)
      {
	 compute_major[device]=5; 
	 compute_minor[device]=2;
      }
      
	
      }

      for (device = 0; device < deviceCount; device ++)
      {
	printf("arch=compute_%d%d,code=sm_%d%d ",compute_major[device],compute_minor[device],sm_major[device],sm_minor[device]);
      }
 
      
     
      return 1; /* failure */
}