/*------------------------------------------------
    Modified from Quantitative TEM/STEM Simulations (QSTEM), author Prof. Christoph Koch.
    url: http://elim.physik.uni-ulm.de/?page_id=834
------------------------------------------------*/

#ifndef RWQSC
#define RWQSC
#include <stdlib.h>
#include <stdio.h>
#include"paramStructure.h"
#include <string.h>
#include <typeinfo>   // operator typeid

#ifdef __linux__
#include <stdbool.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif
 
#include <malloc.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include "cuda_assert.hpp"
#include "memory_fftw3.h"	/* memory allocation routines */
#include "readparams.h"
#include "imagelib_fftw3.h"
#include "fileio_fftw3.h"
#include "data_containers.h"
#include "stemtypes_fftw3.h"

#define NCINMAX 1024
#define NPARAM 64
#define PHASE_GRATING 0
#define BUF_LEN 256

 
#define RAD2DEG 57.2958
// #define SQRT_2 1.4142135
// extern int fftMeasureFlag;





 
extern bool atomsFromExternal;
extern char *elTable;


double wavelength( double kev );
int ReadLine( FILE* fpRead, char* cRead, int cMax, const char *mesg );
 
 




bool readQsc(const char* filename,  params_t** params,int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d);
void initMuls( MULS *muls);
void readSFactLUT( MULS *muls);
void readArray(const char *title,double *array,int N);
#endif