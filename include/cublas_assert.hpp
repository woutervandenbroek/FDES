#ifndef _SDAFION4ELKJANSFD9O4UHKASFDJN948SFDAJKSFAKDJ489SFDA98YSDKDJHASKLFDJH43UBFDI
#define _SDAFION4ELKJANSFD9O4UHKASFDJN948SFDAJKSFAKDJ489SFDA98YSDKDJHASKLFDJH43UBFDI

#include <cublas_v2.h>

#include <cstdio>
#include <stdio.h>

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
        printf( "%s:%lu: cuda runtime error occured:\n[[ERROR]]: %s\n", file, line, error_msg( result ) );
        abort();
    }

  const  char* error_msg( const cublasStatus_t& result ) const
    {
       const char * unkonwnerror ="CUBLAS: an unknown internal error has occurred." ; 
       const char * errorChar;
        if ( result == CUBLAS_STATUS_NOT_INITIALIZED ) {
	  errorChar = "The CUBLAS library was not initialized.  This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the CUCUBLAS routine, or an error in the hardware setup.";
	  return errorChar; }

        if ( result == CUBLAS_STATUS_ALLOC_FAILED ) { errorChar = "Resource allocation failed inside the CUBLAS library. This is usually caused by a cudaMalloc() failure."; 
	  return errorChar;}

        if ( result == CUBLAS_STATUS_INVALID_VALUE ) { errorChar = "An unsupported value or parameter was passed to the function (a negative vector size, for example).";  return errorChar;}

        if ( result == CUBLAS_STATUS_ARCH_MISMATCH ) {  errorChar =  "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision."; return errorChar; }

        if ( result == CUBLAS_STATUS_MAPPING_ERROR ) {  errorChar = "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";return errorChar; }

        if ( result == CUBLAS_STATUS_EXECUTION_FAILED ) { errorChar ="The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons."; return errorChar;}

        if ( result == CUBLAS_STATUS_INTERNAL_ERROR ) { errorChar ="An internal CUBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure."; return errorChar;}

        return unkonwnerror;
    }
};//struct cublas_result_assert 

#define cublas_assert(result) cublas_result_assert()(result, __FILE__, __LINE__)

#endif//NDEBUG

#endif//_SDAFION4ELKJANSFD9O4UHKASFDJN948SFDAJKSFAKDJ489SFDA98YSDKDJHASKLFDJH43UBFDI