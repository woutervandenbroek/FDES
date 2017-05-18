#ifndef _CUDA_ASSERT_HPP_INCLUDED_OPINAFSDLKJNSAFLKJNSADFLJNSAFDLKJNSFDLKJANSFFD
#define _CUDA_ASSERT_HPP_INCLUDED_OPINAFSDLKJNSAFLKJNSADFLJNSAFDLKJNSFDLKJANSFFD

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef cuda_assert
#undef cuda_assert
#endif

#ifdef NDEBUG

#define cuda_assert(e) ((void)0)

#else

extern "C"int printf( const char* __restrict, ... );

struct cuda_result_assert
{
    void operator()( const cudaError_t& result, const char* const file, const unsigned long line ) const
    {
        if ( cudaSuccess == result ) { return; }

        report_error( result, file, line );
    }

    void report_error( const cudaError_t& result, const char* const file, const unsigned long line ) const
    {
        printf( "%s:%u: cuda runtime error occured:\n[[ERROR]]: %s\n", file, line, error_msg( result ) );
        abort();
    }

    char* error_msg( const cudaError_t& result ) const
    {
        if ( result == cudaErrorMissingConfiguration ) { return "The device function being invoked (usually via cudaLaunch()) was not previously configured via the cudaConfigureCall() function."; }

        if ( result == cudaErrorMemoryAllocation ) { return "The API call failed because it was unable to allocate enough memory to perform the requested operation."; }

        if ( result == cudaErrorInitializationError ) { return "The API call failed because the CUDA driver and runtime could not be initialized."; }

        if ( result == cudaErrorLaunchFailure ) { return "An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. The device cannot be used until cudaThreadExit() is called. All existing device memory allocations are invalid and must be reconstructed if the program is to continue using CUDA."; }

        if ( result == cudaErrorPriorLaunchFailure ) { return "  This indicated that a previous kernel launch failed. This was previously used for device emulation of kernel launches."; }

        if ( result == cudaErrorLaunchTimeout ) { return "This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device property kernelExecTimeoutEnabled for more information. The device cannot be used until cudaThreadExit() is called. All existing device memory allocations are invalid and must be reconstructed if the program is to continue using CUDA."; }

        if ( result == cudaErrorLaunchOutOfResources ) { return "This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count."; }

        if ( result == cudaErrorInvalidDeviceFunction ) { return "The requested device function does not exist or is not compiled for the proper device architecture."; }

        if ( result == cudaErrorInvalidConfiguration ) { return "This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See cudaDeviceProp for more device limitations."; }

        if ( result == cudaErrorInvalidDevice ) { return "This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device."; }

        if ( result == cudaErrorInvalidValue ) { return "This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values."; }

        if ( result == cudaErrorInvalidPitchValue ) { return "This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch."; }

        if ( result == cudaErrorInvalidSymbol ) { return "This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier."; }

        if ( result == cudaErrorMapBufferObjectFailed ) { return "This indicates that the buffer object could not be mapped."; }

        if ( result == cudaErrorUnmapBufferObjectFailed ) { return "This indicates that the buffer object could not be unmapped."; }

        if ( result == cudaErrorInvalidHostPointer ) { return "This indicates that at least one host pointer passed to the API call is not a valid host pointer."; }

        if ( result == cudaErrorInvalidDevicePointer ) { return "This indicates that at least one device pointer passed to the API call is not a valid device pointer."; }

        if ( result == cudaErrorInvalidTexture ) { return "This indicates that the texture passed to the API call is not a valid texture."; }

        if ( result == cudaErrorInvalidTextureBinding ) { return "This indicates that the texture binding is not valid. This occurs if you call cudaGetTextureAlignmentOffset() with an unbound texture."; }

        if ( result == cudaErrorInvalidChannelDescriptor ) { return "This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by cudaChannelFormatKind, or if one of the dimensions is invalid."; }

        if ( result == cudaErrorInvalidMemcpyDirection ) { return "This indicates that the direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind."; }

        if ( result == cudaErrorAddressOfConstant ) { return "This indicated that the user has taken the address of a constant variable, which was forbidden up until the CUDA 3.1 release."; }

        if ( result == cudaErrorTextureFetchFailed ) { return "This indicated that a texture fetch was not able to be performed. This was previously used for device emulation of texture operations."; }

        if ( result == cudaErrorTextureNotBound ) { return "This indicated that a texture was not bound for access. This was previously used for device emulation of texture operations."; }

        if ( result == cudaErrorSynchronizationError ) { return "This indicated that a synchronization operation had failed. This was previously used for some device emulation functions."; }

        if ( result == cudaErrorInvalidFilterSetting ) { return "This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA."; }

        if ( result == cudaErrorInvalidNormSetting ) { return "This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA."; }

        if ( result == cudaErrorMixedDeviceExecution ) { return "Mixing of device and device emulation code was not allowed."; }

        if ( result == cudaErrorCudartUnloading ) { return "This indicated an issue with calling API functions during the unload process of the CUDA runtime in prior releases."; }

        if ( result == cudaErrorUnknown ) { return "This indicates that an unknown internal error has occurred."; }

        if ( result == cudaErrorNotYetImplemented ) { return "This indicates that the API call is not yet implemented. Production releases of CUDA will never return this error."; }

        if ( result == cudaErrorMemoryValueTooLarge ) { return "This indicated that an emulated device pointer exceeded the 32-bit address range."; }

        if ( result == cudaErrorInvalidResourceHandle ) { return "This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like cudaStream_t and cudaEvent_t."; }

        if ( result == cudaErrorNotReady ) { return "This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than cudaSuccess (which indicates completion). Calls that may return this value include cudaEventQuery() and cudaStreamQuery()."; }

        if ( result == cudaErrorInsufficientDriver ) { return "This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run."; }

        if ( result == cudaErrorSetOnActiveProcess ) { return "This indicates that the user has called cudaSetDevice(), cudaSetValidDevices(), cudaSetDeviceFlags(), cudaD3D9SetDirect3DDevice(), cudaD3D10SetDirect3DDevice, cudaD3D11SetDirect3DDevice(), * or cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by calling non-device management operations (allocating memory and launching kernels are examples of non-device management operations). This error can also be returned if using runtime/driver interoperability and there is an existing CUcontext active on the host thread."; }

        if ( result == cudaErrorInvalidSurface ) { return "This indicates that the surface passed to the API call is not a valid surface."; }

        if ( result == cudaErrorNoDevice ) { return "This indicates that no CUDA-capable devices were detected by the installed CUDA driver."; }

        if ( result == cudaErrorECCUncorrectable ) { return "This indicates that an uncorrectable ECC error was detected during execution."; }

        if ( result == cudaErrorSharedObjectSymbolNotFound ) { return "This indicates that a link to a shared object failed to resolve."; }

        if ( result == cudaErrorSharedObjectInitFailed ) { return "This indicates that initialization of a shared object failed."; }

        if ( result == cudaErrorUnsupportedLimit ) { return "This indicates that the cudaLimit passed to the API call is not supported by the active device."; }

        if ( result == cudaErrorDuplicateVariableName ) { return "This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name."; }

        if ( result == cudaErrorDuplicateTextureName ) { return "This indicates that multiple textures (across separate CUDA source files in the application) share the same string name."; }

        if ( result == cudaErrorDuplicateSurfaceName ) { return "This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name."; }

        if ( result == cudaErrorDevicesUnavailable ) { return "This indicates that all CUDA devices are busy or unavailable at the current time. Devices are often busy/unavailable due to use of cudaComputeModeExclusive, cudaComputeModeProhibited or when long running CUDA kernels have filled up the GPU and are blocking new work from starting. They can also be unavailable due to memory constraints on a device that already has active CUDA work being performed."; }

        if ( result == cudaErrorInvalidKernelImage ) { return "This indicates that the device kernel image is invalid."; }

        if ( result == cudaErrorNoKernelImageForDevice ) { return "This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration."; }

        if ( result == cudaErrorIncompatibleDriverContext ) { return "This indicates that the current context is not compatible with this the CUDA Runtime. This can only occur if you are using CUDA Runtime/Driver interoperability and have created an existing Driver context using the driver API. The Driver context may be incompatible either because the Driver context was created using an older version of the API, because the Runtime API call expects a primary driver contextand the Driver context is not primary, or because the Driver context has been destroyed. Please see Interactions with the CUDA Driver API for more information."; }

        if ( result == cudaErrorPeerAccessAlreadyEnabled ) { return "This error indicates that a call to cudaDeviceEnablePeerAccess() is trying to re - enable peer addressing on from a context which has already had peer addressing enabled."; }

        if ( result == cudaErrorPeerAccessNotEnabled ) { return "This error indicates that cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via cudaDeviceEnablePeerAccess()."; }

        if ( result == cudaErrorDeviceAlreadyInUse ) { return "This indicates that a call tried to access an exclusive - thread device that is already in use by a different thread."; }

        if ( result == cudaErrorProfilerDisabled ) { return "This indicates profiler has been disabled for this run and thus runtime APIs cannot be used to profile subsets of the program. This can happen when the application is running with external profiling tools like visual profiler."; }

        if ( result == cudaErrorProfilerNotInitialized ) { return "This indicates profiler has not been initialized yet. cudaProfilerInitialize() must be called before calling cudaProfilerStart and cudaProfilerStop to initialize profiler."; }

        if ( result == cudaErrorProfilerAlreadyStarted ) { return "This indicates profiler is already started. This error can be returned if cudaProfilerStart() is called multiple times without subsequent call to cudaProfilerStop()."; }

        if ( result == cudaErrorProfilerAlreadyStopped ) { return "This indicates profiler is already stopped. This error can be returned if cudaProfilerStop() is called without starting profiler using cudaProfilerStart()."; }

        if ( result == cudaErrorStartupFailure ) { return "This indicates an internal startup failure in the CUDA runtime."; }

        if ( result == cudaErrorApiFailureBase ) { return "Any unhandled CUDA driver error is added to this value and returned via the runtime. Production releases of CUDA should not return such errors. "; }

        return "CUDA: an unknown internal error has occurred.";
    }
};//struct cuda_result_assert 

#define cuda_assert(result) cuda_result_assert()(result, __FILE__, __LINE__)

#endif//NDEBUG

#endif//_CUDA_ASSERT_HPP_INCLUDED_OPINAFSDLKJNSAFLKJNSADFLJNSAFDLKJNSFDLKJANSFFD 