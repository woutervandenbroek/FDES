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


#ifndef curand_assert_lizgelkuzgqwozhblhbqwerqzfwhgrerwoi
#define curand_assert_lizgelkuzgqwozhblhbqwerqzfwhgrerwoi

#include <curand.h>
#include <cstdlib>
#include <stdio.h>

#ifdef curand_assert
#undef curand_assert
#endif

#ifdef NDEBUG

#define curand_assert(e) ((void)0)

#else



struct curand_result_assert
{
    void operator()( const curandStatus_t& result, const char* const file, const unsigned long line ) const
    {
        if ( CURAND_STATUS_SUCCESS == result ) { return; }

        report_error( result, file, line );
    }

    void report_error( const curandStatus_t& result, const char* const file, const unsigned long line ) const
    {
        printf( "%s:%lu: cuda runtime error occured:\n[[ERROR]]: %s\n", file, line, error_msg( result ) );
        abort();
    }

   const char* error_msg( const curandStatus_t& result ) const
    {
		const char * unkonwnerror ="CURAND: an unknown internal error has occurred." ;
		
		const char * errorChar;
		
		if ( result == CURAND_STATUS_VERSION_MISMATCH ) { errorChar = "CURAND: Header le and linked library version do not match."; return errorChar; }

		if ( result == CURAND_STATUS_NOT_INITIALIZED ) { errorChar = "CURAND: Generator not initialized.";return errorChar;  }

		if ( result == CURAND_STATUS_ALLOCATION_FAILED ) { errorChar = "CURAND: Memory allocation failed.";return errorChar;  }

		if ( result == CURAND_STATUS_TYPE_ERROR ) { errorChar = "CURAND: Generator is wrong type."; return errorChar; }

		if ( result == CURAND_STATUS_OUT_OF_RANGE ) { errorChar = "CURAND: Argument out of range."; return errorChar; }

		if ( result == CURAND_STATUS_LENGTH_NOT_MULTIPLE ) { errorChar = "CURAND: Length requested is not a multiple of dimension.";return errorChar;  }

		if ( result == CURAND_STATUS_DOUBLE_PRECISION_REQUIRED ) { errorChar = "CURAND: GPU does not have double precision required by MRG32k3a.";return errorChar;  }

		if ( result == CURAND_STATUS_LAUNCH_FAILURE ) { errorChar = "CURAND: Kernel launch failure."; }

		if ( result == CURAND_STATUS_PREEXISTING_FAILURE ) { errorChar = "CURAND: Preexisting failure on library entry.";return errorChar;  }

		if ( result == CURAND_STATUS_INITIALIZATION_FAILED ) { errorChar = "CURAND: Initialization of CUDA failed.";return errorChar;  }

		if ( result == CURAND_STATUS_ARCH_MISMATCH ) { errorChar = "CURAND: Architecture mismatch, GPU does not support requested feature.";return errorChar;  }

		if ( result == CURAND_STATUS_INTERNAL_ERROR ) { errorChar = "CURAND: Internal library error."; return errorChar; }

		return unkonwnerror;
    }
};//struct curand_result_assert 

#define curand_assert(result) curand_result_assert()(result, __FILE__, __LINE__)

#endif//NDEBUG

#endif//curand_assert_lizgelkuzgqwozhblhbqwerqzfwhgrerwoi 