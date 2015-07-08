cmake_minimum_required(VERSION 2.4)
project(qstem_libs)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_BUILD_TYPE Release)

find_package(fftw3 REQUIRED)
find_package(Boost REQUIRED)
include_directories(${FFTW3_INCLUDE_DIRS} ${FFTW3_LIBS} ${FFTW3F_LIBS})
include_directories(${Boost_INCLUDE_DIR})
link_directories(${FFTW3_INCLUDE_DIRS}) 

if(WIN32)
IF(CMAKE_BUILD_TYPE MATCHES Debug) 
   link_directories(${CMAKE_SOURCE_DIR}/Debug)
 ELSEIF(CMAKE_BUILD_TYPE MATCHES Release)
   link_directories(${CMAKE_SOURCE_DIR}/Release)
 ELSE()
   link_directories(${CMAKE_SOURCE_DIR}/Debug)
ENDIF()
endif()

FILE(GLOB STEM3_LIBS_C_FILES "${CMAKE_SOURCE_DIR}/*.cpp")
FILE(GLOB STEM3_LIBS_H_FILES "${CMAKE_SOURCE_DIR}/*.h")
set (qstem_libs_src ${STEM3_LIBS_C_FILES} ${STEM3_LIBS_H_FILES})

if(WIN32)
link_libraries(qstem_libs libfftw3-3 libfftw3l-3 libfftw3f-3)
endif()
add_library(qstem_libs_static STATIC ${qstem_libs_src})

SET_TARGET_PROPERTIES(qstem_libs_static
  PROPERTIES OUTPUT_NAME qstem_libs
  CLEAN_DIRECT_OUTPUT 1)
  
add_library(qstem_libs_shared SHARED ${qstem_libs_src})

SET_TARGET_PROPERTIES(qstem_libs_shared
  PROPERTIES OUTPUT_NAME qstem_libs
  CLEAN_DIRECT_OUTPUT 1)

if(WIN32)
IF(CMAKE_BUILD_TYPE MATCHES Debug) 
    set(lib_src Debug/qstem_libs.lib)
    set(dll_src Debug/qstem_libs.dll)
 ELSEIF(CMAKE_BUILD_TYPE MATCHES Release)
    set(lib_src Release/qstem_libs.lib)
    set(dll_src Release/qstem_libs.dll )
 ELSE()
    set(lib_src Debug/qstem_libs.lib)
    set(dll_src Debug/qstem_libs.dll)
ENDIF()
ADD_CUSTOM_COMMAND(TARGET qstem_libs_static
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${lib_src} ${CMAKE_SOURCE_DIR}
      )
ADD_CUSTOM_COMMAND(TARGET qstem_libs_shared
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${dll_src} ${CMAKE_SOURCE_DIR}
)
endif()

