#ifndef _CUFFT_ASSERT_HPP_INCLUDED_LKJASHGLAQIURHGAFKJNGWQASKJDHAQWIRT
#define _CUFFT_ASSERT_HPP_INCLUDED_LKJASHGLAQIURHGAFKJNGWQASKJDHAQWIRT

#include <cufft.h>
#include <cstdlib>

#ifdef cufft_assert
#undef cufft_assert
#endif

#ifdef NDEBUG

#define cufft_assert(e) ((void)0)

#else

extern "C"int printf( const char* __restrict, ... );

struct cufft_result_assert
{
    void operator()( const cufftResult_t& result, const char* const file, const unsigned long line ) const
    {
        if ( CUFFT_SUCCESS == result ) { return; }

        report_error( result, file, line );
    }

    void report_error( const cufftResult_t& result, const char* const file, const unsigned long line ) const
    {
        printf( "%s:%u: cuda runtime error occured:\n[[ERROR]]: %s\n", file, line, error_msg( result ) );
        abort();
    }

    char* error_msg( const cufftResult_t& result ) const
    {
        if ( result == CUFFT_INVALID_PLAN ) { return "CUFFT: Invalid plan."; }

        if ( result == CUFFT_ALLOC_FAILED ) { return "CUFFT: Allocation failed."; }

        if ( result == CUFFT_INVALID_TYPE ) { return "CUFFT: Invalid type."; }

        if ( result == CUFFT_INVALID_VALUE ) { return "CUFFT: Invalid value."; }

        if ( result == CUFFT_INTERNAL_ERROR ) { return "CUFFT: Internal error."; }

        if ( result == CUFFT_EXEC_FAILED ) { return "CUFFT: Execution failed."; }

        if ( result == CUFFT_SETUP_FAILED ) { return "CUFFT: Setup failed."; }

        if ( result == CUFFT_INVALID_SIZE ) { return "CUFFT: Invalid size."; }

        if ( result == CUFFT_UNALIGNED_DATA ) { return "CUFFT: Unaligned data."; }

        return "CUFFT: an unknown internal error has occurred.";
    }
};//struct cufft_result_assert 

#define cufft_assert(result) cufft_result_assert()(result, __FILE__, __LINE__)

#endif//NDEBUG

#endif//_CUFFT_ASSERT_HPP_INCLUDED_LKJASHGLAQIURHGAFKJNGWQASKJDHAQWIRT 