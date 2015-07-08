#ifndef _SDF9P4JHIELKJAFD2349U8YHSFDAKJLASFDKJBNCVMNBASDFKJBHSAFDBH4IUASFDJH3FSD
#define _SDF9P4JHIELKJAFD2349U8YHSFDAKJLASFDKJBNCVMNBASDFKJBHSAFDBH4IUASFDJH3FSD

#include "cuda_assert.hpp"

#ifdef kernel_assert
#undef kernel_assert 
#endif

#define kernel_assert(x) \
        do \
        { \
            x; \
            cuda_assert( cudaGetLastError() ); \
        } \
        while(0)

#endif//_SDF9P4JHIELKJAFD2349U8YHSFDAKJLASFDKJBNCVMNBASDFKJBHSAFDBH4IUASFDJH3FSD

