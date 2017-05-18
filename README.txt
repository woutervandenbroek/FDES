
FDES, forward dynamical electron scattering, 
is a GPU-based multislice algorithm.

====================================================================

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

Associated article: 
W. Van den Broek, X. Jiang, C.T. Koch. FDES, a GPU-based multislice 
algorithm with increased effciency of the computation of the 
projected potential. Ultramicroscopy (2015). 
doi:10.1016/j.ultramic.2015.07.005

Email: wouter.vandenbroek@uni-ulm.de, wouter.vandenbroek1@gmail.com,
       xiaoming.jiang@uni-ulm.de, jiang.xiaoming1984@gmail.com

Address: Institut for Experimentel Physics
         Ulm University
         Albert-Einstein-Allee 11
         89081 Ulm
         Germany

====================================================================

             
  Steps of FDES building
  
  0. For the FDES light version the libraries boost, hdf5 and qstem-libs are not required. The program only reads input from the text file "Params.cnf" and writes the result as raw 8-bit floats under the name "Measurements.bin".
  
  1. install required libraries with suggested versions: cmake 2.8, fftw 3.2.2, boost 1.53, hdf5 1.8.11, CUDA 5.0 
  
  I.  Linux system
  
  sudo apt-get install cmake 
  
  sudo apt-get install libfftw3-dev 
  
  sudo apt-get install libboost-all-dev
  
  sudo apt-get install libhdf5-dev
  
  Installation of CUDA under linux could be referred through the following link:
  http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz3LPMuI9c3
    
  II. Windows system:
  
  Notice: FDES is tested using Microsoft Visual Studio (VS) 2008. For the higher versions of VS cmake package might not support the integration between CUDA and VS. In this case users might compile the source code with manual configuration instead of using cmake.
  
  Cmake: http://www.cmake.org/download/
  
  Boost: pre-compiled versions from  http://boost.teeks99.com/ or directly from http://www.boost.org/doc/libs/1_55_0/more/getting_started/windows.html
  
  HDF5:  http://www.hdfgroup.org/HDF5/release/obtain5.html
  
  FFTW3: http://www.fftw.org/install/windows.html
  
  Installation of CUDA under windows could be referred through the following link:
  http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/#axzz3LPMuI9c3
 
  

  
  2. build source files
  
  I. Linux system
  
  a. build qstem-libs
  enter "qstem-libs" folder:  cd /path/to/FDES/qstem-libs 
  run cmake: cmake .
  run makefile:  make
  
  b. build FDES
  enter "FDES" folder: cd /path/to/FDES
  run cmake:  cmake .
  run makefile: make
  
  the compiled program "FDES" could be found in "bin" folder.
  
  II. windows system
  
  a. build qstem-libs
  
  run cmake-gui
  set path "where is the source code" to be the path to the folder "qstem-libs", for example, "C:/Users/admin/FDES/qstem-libs".

  set path "where to build the binaries" to be the same path as the source code path.
  
  click the button "Configure" and choose the specified VS version (VS 2008 or previous versions). Errors would be reported if there are no recognizable boost or fftw3 packages in windows system. Errors could be solved by manual filling the following cmake variables: Boost_INCLUDE_DIR, FFTW3_INCLUDE_DIR, FFTW3_LIBS, FFTW3F_LIBS.
  
  click the button "Configure" again until there is no reported error. 
  
  click the button "Generate" to generate the project file.
  
  enter the folder "qstem-libs". Open "ALL_BUILD.vcproj" file and build the project. "qstem_libs.lib" and "qstem_libs.dll" would appear in this folder after successful compiling operations.
  
  b. build FDES
  
  run a new cmake-gui
  set path "where is the source code" to be the path to the folder "FDES", for example, "C:/Users/admin/FDES".

  set path "where to build the binaries" to be the same path as the source code path.
  
  click the button "Configure" and choose the specified VS version (VS 2008 or previous versions). Errors would be reported if there are no recognizable boost, fftw3 hdf5 or cuda packages in windows system. Errors could be solved manual filling the following cmake variables: Boost_INCLUDE_DIR, FFTW3_INCLUDE_DIR, FFTW3_LIBS, FFTW3F_LIBS, HDF5_C_INCLUDE_DIR and HDF5_hdf5_LIBRARY_RELEASE(DEBUG). Variables for cuda package could be generally detected. If not, variables CUDA_CUDART_LIBRARY, CUDA_CUDA_LIBRARY,  CUDA_NVCC_EXECUTABLE, CUDA_TOOLKIT_INCLUDE, CUDA_cublas_LIBRARY, CUDA_cufft_LIBRARY should be at least specified.
  
  click the button "Configure" again until there is no reported error. 
  
  click the button "Generate" to generate the project file.
  
  enter the folder "FDES". Open "ALL_BUILD.vcproj" file and build the project. "FDES.exe" would appear in "bin" folder after successful compiling operations.
   
  
  3. Possible errors and notice:
  
  Linux
  1. " error: identifier "__float128" is undefined "  This could be solved by changing fftw3.h from

  /* __float128 (quad precision) is a gcc extension on i386, x86_64, and ia64     
   for gcc >= 4.6 (compiled in FFTW with --enable-quad-precision) */
  #if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)) \
  && !(defined(__ICC) || defined(__INTEL_COMPILER)) \
  && (defined(__i386__) || defined(__x86_64__) || defined(__ia64__))
 
  to

  /* __float128 (quad precision) is a gcc extension on i386, x86_64, and ia64     
   for gcc >= 4.6 (compiled in FFTW with --enable-quad-precision) */
  #if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)) \
  && !(defined(__ICC) || defined(__INTEL_COMPILER) || defined(__CUDACC__)) \
  && (defined(__i386__) || defined(__x86_64__) || defined(__ia64__))
  
  2. "error while loading shared libraries: XX.so: cannot open shared object file: No such file or directory" 
  This error indicates the excutable is not linked to the shared library. Add the path containing XX.so file to the  environment variable LD_LIBRARY_PATH  by excuting the command:   	
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/XX
  
 
  Windows 
  1.  "cmd.exe" exited with code 2 under windows
  This means "cmd.exe" is executing an unrecognized command. As mentioned before, downgrade VS to the version of 2008 or 2005.
  
  2. "LINK : fatal error LNK1104: cannot open file '.\CMakeFiles\FDES.dir\src\$(OutDir)\FDES_generated_complexMath.cu.obj'" 
  Try to change the cmake to the version 2.8 
  
  3. "fatal error C1083: Cannot open include file: 'stdint.h': No such file or directory"
  stdint.h file is not included in VS 2008 and previous versions.  Download a MS version of this header from: http://msinttypes.googlecode.com/svn/trunk/stdint.h and copy it to VS directory. For example: C:\Program Files\Microsoft Visual Studio 9.0\VC\include
  
  4. switch between debug and release mode (either for linux or windows)
  Open "CMakeLists.txt" file and comment or uncommnet the proper line,  "set(CMAKE_BUILD_TYPE Release)" or "set(CMAKE_BUILD_TYPE Debug)". And then run cmake again to generate a new project.
  
  5. "The program can't start because XX.dll is missing from your computer." 
  This error could be solved by placing the missing dll files into the same folder or manually setting the project environment to the correct path.

  
  6. python support
  FDES kernal can be executed from python command by the use of ctypes module. A detailed example, pyFDES.py file, could be found in the "python" sub-folder. Notice this example is verified under linux system, and for windows users it might report some unkown errors.
  
  
  
  
  
  
  
  
  

 