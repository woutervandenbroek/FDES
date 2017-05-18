#ifndef curand_assert_lizgelkuzgqwozhblhbqwerqzfwhgrerwoi
#define curand_assert_lizgelkuzgqwozhblhbqwerqzfwhgrerwoi

#include <curand.h>
#include <cstdlib>

#ifdef curand_assert
#undef curand_assert
#endif

#ifdef NDEBUG

#define curand_assert(e) ((void)0)

#else

extern "C"int printf( const char* __restrict, ... );

struct curand_result_assert
{
    void operator()( const curandStatus_t& result, const char* const file, const unsigned long line ) const
    {
        if ( CURAND_STATUS_SUCCESS == result ) { return; }

        report_error( result, file, line );
    }

    void report_error( const curandStatus_t& result, const char* const file, const unsigned long line ) const
    {
        printf( "%s:%u: cuda runtime error occured:\n[[ERROR]]: %s\n", file, line, error_msg( result ) );
        abort();
    }

    char* error_msg( const curandStatus_t& result ) const
    {
		if ( result == CURAND_STATUS_VERSION_MISMATCH ) { return "CURAND: Header le and linked library version do not match."; }

		if ( result == CURAND_STATUS_NOT_INITIALIZED ) { return "CURAND: Generator not initialized."; }

		if ( result == CURAND_STATUS_ALLOCATION_FAILED ) { return "CURAND: Memory allocation failed."; }

		if ( result == CURAND_STATUS_TYPE_ERROR ) { return "CURAND: Generator is wrong type."; }

		if ( result == CURAND_STATUS_OUT_OF_RANGE ) { return "CURAND: Argument out of range."; }

		if ( result == CURAND_STATUS_LENGTH_NOT_MULTIPLE ) { return "CURAND: Length requested is not a multiple of dimension."; }

		if ( result == CURAND_STATUS_DOUBLE_PRECISION_REQUIRED ) { return "CURAND: GPU does not have double precision required by MRG32k3a."; }

		if ( result == CURAND_STATUS_LAUNCH_FAILURE ) { return "CURAND: Kernel launch failure."; }

		if ( result == CURAND_STATUS_PREEXISTING_FAILURE ) { return "CURAND: Preexisting failure on library entry."; }

		if ( result == CURAND_STATUS_INITIALIZATION_FAILED ) { return "CURAND: Initialization of CUDA failed."; }

		if ( result == CURAND_STATUS_ARCH_MISMATCH ) { return "CURAND: Architecture mismatch, GPU does not support requested feature."; }

		if ( result == CURAND_STATUS_INTERNAL_ERROR ) { return "CURAND: Internal library error."; }

		return "CURAND: an unknown internal error has occurred.";
    }
};//struct curand_result_assert 

#define curand_assert(result) curand_result_assert()(result, __FILE__, __LINE__)

#endif//NDEBUG

#endif//curand_assert_lizgelkuzgqwozhblhbqwerqzfwhgrerwoi 