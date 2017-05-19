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

project(FDES)
set(CMAKE_BUILD_TYPE Release)# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDA REQUIRED)

include(${CMAKE_MODULE_PATH}/CheckComputeCapability.cmake)
string(REPLACE " " ";" CUDA_CAPABILITY_LIST ${CUDA_CAPABILITY_INFO})

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
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIE;)             
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
foreach(cuda_gpu ${CUDA_CAPABILITY_LIST}) 
list(APPEND CUDA_NVCC_FLAGS "-gencode ${cuda_gpu}")
endforeach(cuda_gpu)


FILE(GLOB FDES_CU_FILES "${CMAKE_SOURCE_DIR}/src/*.cu""${CMAKE_SOURCE_DIR}/src/*.cpp")
FILE(GLOB FDES_H_FILES "${CMAKE_SOURCE_DIR}/include/*.h")
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_TOOLKIT_INCLUDE})


CUDA_ADD_EXECUTABLE(FDES ${FDES_CU_FILES})

if (WIN32)
TARGET_LINK_LIBRARIES(FDES "cudart" "cufft" "cublas")
endif (WIN32)

if(UNIX)
TARGET_LINK_LIBRARIES(FDES ${CUDA_CUDART_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_cufft_LIBRARY})
endif(UNIX)


