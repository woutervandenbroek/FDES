#ifndef _SDAFION4ELKJANSFD9O4UHKASFDJN948SFDAJKSFAKDJ489SFDA98YSDKDJHASKLFDJH43UBFDI
#define _SDAFION4ELKJANSFD9O4UHKASFDJN948SFDAJKSFAKDJ489SFDA98YSDKDJHASKLFDJH43UBFDI

#include <cublas_v2.h>

#include <cstdio>

#ifdef cublas_assert
#undef cublas_assert
#endif

#ifdef NDEBUG

#define cublas_assert(e) ((void)0)

#else

struct cublas_result_assert
{
    void operator()( const cublasStatus_t& result, const char* const file, const unsigned long line ) const
    {
        if ( CUBLAS_STATUS_SUCCESS == result ) { return; }

        report_error( result, file, line );
    }

    void report_error( const cublasStatus_t& result, const char* const file, const unsigned long line ) const
    {
        printf( "%s:%u: cuda runtime error occured:\n[[ERROR]]: %s\n", file, line, error_msg( result ) );
        abort();
    }

    char* error_msg( const cublasStatus_t& result ) const
    {
        if ( result == CUBLAS_STATUS_NOT_INITIALIZED ) { return "The CUBLAS library was not initialized.  This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the CUCUBLAS routine, or an error in the hardware setup."; }

        if ( result == CUBLAS_STATUS_ALLOC_FAILED ) { return "Resource allocation failed inside the CUBLAS library. This is usually caused by a cudaMalloc() failure."; }

        if ( result == CUBLAS_STATUS_INVALID_VALUE ) { return "An unsupported value or parameter was passed to the function (a negative vector size, for example)."; }

        if ( result == CUBLAS_STATUS_ARCH_MISMATCH ) { return "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision."; }

        if ( result == CUBLAS_STATUS_MAPPING_ERROR ) { return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture."; }

        if ( result == CUBLAS_STATUS_EXECUTION_FAILED ) { return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons."; }

        if ( result == CUBLAS_STATUS_INTERNAL_ERROR ) { return "An internal CUBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure."; }

        return "CUBLAS: an unknown internal error has occurred.";
    }
};//struct cublas_result_assert 

#define cublas_assert(result) cublas_result_assert()(result, __FILE__, __LINE__)

#endif//NDEBUG

#endif//_SDAFION4ELKJANSFD9O4UHKASFDJN948SFDAJKSFAKDJ489SFDA98YSDKDJHASKLFDJH43UBFDI