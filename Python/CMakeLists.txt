# ====================================================================
#
# Copyright (C) 2015 Wouter Van den Broek, Xiaoming Jiang
# 
# This file is part of FDES.
# 
# FDES is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# FDES is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with FDES. If not, see <http://www.gnu.org/licenses/>.
# 
# Email: wouter.vandenbroek@uni-ulm.de, wouter.vandenbroek1@gmail.com,
#        xiaoming.jiang@uni-ulm.de, jiang.xiaoming1984@gmail.com
# 
# ====================================================================


cmake_minimum_required(VERSION 2.4)

project(FDES_SHARED_LIB_SHARED_LIB)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
if (CMAKE_BUILD_TYPE MATCHES Release)
        if (MSVC)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/EHsc;)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/MD;)
	    list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/W3;)
	    list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/nologo;)
	    list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/O2;)
	    list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/Zi;)
        else ()
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-O2;)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-pipe;)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-mtune=generic;)   
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fstack-protector;) 
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wall;) 
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-W;) 
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-D_REENTRANT;)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC;)             
        endif ()
else ()
        if (MSVC)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/EHsc;)
	    list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/W3;)
	    list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/nologo;)
	    list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/Od;)
	    list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/Zi;)
	    list(APPEND CUDA_NVCC_FLAGS -Xcompiler;/MDd;)
        else ()
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-O0;)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-g;)
        endif ()
endif ()

FILE(GLOB FDES_SHARED_LIB_CU_FILES "${CMAKE_SOURCE_DIR}/src/complexMath.cu" "${CMAKE_SOURCE_DIR}/src/crystalMaker.cu" "${CMAKE_SOURCE_DIR}/src/FDESExport.cu" "${CMAKE_SOURCE_DIR}/src/multisliceSimulation.cu" "${CMAKE_SOURCE_DIR}/src/optimFunctions.cu"  "${CMAKE_SOURCE_DIR}/src/paramStructure.cu"  "${CMAKE_SOURCE_DIR}/src/projectedPotential.cu" "${CMAKE_SOURCE_DIR}/src/rwHdf5.cu"  "${CMAKE_SOURCE_DIR}/src/rwQsc.cu" "${CMAKE_SOURCE_DIR}/src/*.cpp")
FILE(GLOB FDES_SHARED_LIB_H_FILES "${CMAKE_SOURCE_DIR}/include/*.h")
FILE(GLOB LIB_QSTEM_H_FILES "${CMAKE_SOURCE_DIR}/qstem-libs/*.h")
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${HDF5_C_INCLUDE_DIR})
include_directories(${CUDA_TOOLKIT_INCLUDE})
include_directories(${CMAKE_SOURCE_DIR}/qstem-libs)
include_directories(${FFTW3_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})

link_directories(${CMAKE_SOURCE_DIR}/qstem-libs)
link_directories(${FFTW3_LIBS})

IF(CMAKE_BUILD_TYPE MATCHES Debug) 
    link_directories(${HDF5_hdf5_LIBRARY_DEBUG})
 ELSEIF(CMAKE_BUILD_TYPE MATCHES Release)
    link_directories(${HDF5_hdf5_LIBRARY_RELEASE})
 ELSE()
    link_directories(${HDF5_hdf5_LIBRARY_RELEASE})
ENDIF()

cuda_add_library(FDES_SHARED_LIB SHARED ${FDES_SHARED_LIB_CU_FILES})

if (WIN32)
TARGET_LINK_LIBRARIES(FDES_SHARED_LIB "qstem_libs" "cudart" "cufft" "cublas" "libfftw3-3" "hdf5" "libfftw3f-3")
endif (WIN32)

if(UNIX)
TARGET_LINK_LIBRARIES(FDES_SHARED_LIB ${CUDA_CUDART_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_cufft_LIBRARY} ${HDF5_hdf5_LIBRARY_RELEASE} ${CMAKE_SOURCE_DIR}/qstem-libs/libqstem_libs.so ${FFTW3_LIBS} ${FFTW3F_LIBS})
endif(UNIX)
