#ifndef _CUFFT_ASSERT_HPP_INCLUDED_LKJASHGLAQIURHGAFKJNGWQASKJDHAQWIRT
#define _CUFFT_ASSERT_HPP_INCLUDED_LKJASHGLAQIURHGAFKJNGWQASKJDHAQWIRT

#include <cufft.h>
#include <cstdlib>
#include <stdio.h>

#ifdef cufft_assert
#undef cufft_assert
#endif

#ifdef NDEBUG

#define cufft_assert(e) ((void)0)

#else

struct cufft_result_assert
{
    void operator()( const cufftResult_t& result, const char* const file, const unsigned long line ) const
    {
        if ( CUFFT_SUCCESS == result ) { return; }

        report_error( result, file, line );
    }

    void report_error( const cufftResult_t& result, const char* const file, const unsigned long line ) const
    {
        printf( "%s:%lu: cuda runtime error occured:\n[[ERROR]]: %s\n", file, line, error_msg( result ) );
        abort();
    }

   const char* error_msg( const cufftResult_t& result ) const
    {
        const char * unkonwnerror ="CUFFT: an unknown internal error has occurred." ; 
	 const char * errorChar;
        if ( result == CUFFT_INVALID_PLAN ) { errorChar = "CUFFT: Invalid plan."; return errorChar; }

        if ( result == CUFFT_ALLOC_FAILED ) { errorChar = "CUFFT: Allocation failed.";  return errorChar;}

        if ( result == CUFFT_INVALID_TYPE ) { errorChar = "CUFFT: Invalid type.";  return errorChar;}

        if ( result == CUFFT_INVALID_VALUE ) { errorChar = "CUFFT: Invalid value.";  return errorChar;}

        if ( result == CUFFT_INTERNAL_ERROR ) { errorChar = "CUFFT: Internal error.";  return errorChar;}

        if ( result == CUFFT_EXEC_FAILED ) { errorChar = "CUFFT: Execution failed.";  return errorChar;}

        if ( result == CUFFT_SETUP_FAILED ) { errorChar = "CUFFT: Setup failed.";  return errorChar;}

        if ( result == CUFFT_INVALID_SIZE ) { errorChar = "CUFFT: Invalid size.";  return errorChar;}

        if ( result == CUFFT_UNALIGNED_DATA ) { errorChar = "CUFFT: Unaligned data.";  return errorChar;}

        return unkonwnerror;
    }
};//struct cufft_result_assert 

#define cufft_assert(result) cufft_result_assert()(result, __FILE__, __LINE__)

#endif//NDEBUG

#endif//_CUFFT_ASSERT_HPP_INCLUDED_LKJASHGLAQIURHGAFKJNGWQASKJDHAQWIRT 