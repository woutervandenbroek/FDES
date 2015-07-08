#ifndef _CUDA_ASSERT_HPP_INCLUDED_OPINAFSDLKJNSAFLKJNSADFLJNSAFDLKJNSFDLKJANSFFD
#define _CUDA_ASSERT_HPP_INCLUDED_OPINAFSDLKJNSAFLKJNSADFLJNSAFDLKJNSFDLKJANSFFD

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifdef cuda_assert
#undef cuda_assert
#endif

#ifdef NDEBUG

#define cuda_assert(e) ((void)0)

#else

struct cuda_result_assert
{
    void operator()( const cudaError_t& result, const char* const file, const unsigned long line ) const
    {
        if ( cudaSuccess == result ) { return; }

        report_error( result, file, line );
    }

    void report_error( const cudaError_t& result, const char* const file, const unsigned long line ) const
    {
        printf( "%s:%lu: cuda runtime error occured:\n[[ERROR]]: %s\n", file, line, error_msg( result ) );
        abort();
    }

   const char* error_msg( const cudaError_t& result ) const
    {
	const char * unkonwnerror ="CUDA: an unknown internal error has occurred." ; 
	
	const char * errorChar;
       
        if ( result == cudaErrorMissingConfiguration ) { errorChar = "The device function being invoked (usually via cudaLaunch()) was not previously configured via the cudaConfigureCall() function."; 
	  return errorChar; }

        if ( result == cudaErrorMemoryAllocation ) {  errorChar =  "The API call failed because it was unable to allocate enough memory to perform the requested operation.";
	  return errorChar; }

        if ( result == cudaErrorInitializationError ) {  errorChar =  "The API call failed because the CUDA driver and runtime could not be initialized.";
	  return errorChar; }

        if ( result == cudaErrorLaunchFailure ) {  errorChar =  "An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. The device cannot be used until cudaThreadExit() is called. All existing device memory allocations are invalid and must be reconstructed if the program is to continue using CUDA."; 
	  return errorChar;}

        if ( result == cudaErrorPriorLaunchFailure ) {  errorChar =  "  This indicated that a previous kernel launch failed. This was previously used for device emulation of kernel launches."; 
	  return errorChar;}

        if ( result == cudaErrorLaunchTimeout ) {  errorChar =  "This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device property kernelExecTimeoutEnabled for more information. The device cannot be used until cudaThreadExit() is called. All existing device memory allocations are invalid and must be reconstructed if the program is to continue using CUDA."; 
	  return errorChar;}

        if ( result == cudaErrorLaunchOutOfResources ) {  errorChar =  "This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count."; return errorChar;}

        if ( result == cudaErrorInvalidDeviceFunction ) {  errorChar =  "The requested device function does not exist or is not compiled for the proper device architecture.";
	  return errorChar; }

        if ( result == cudaErrorInvalidConfiguration ) {  errorChar =  "This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See cudaDeviceProp for more device limitations."; 
	  return errorChar;}

        if ( result == cudaErrorInvalidDevice ) {  errorChar =  "This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device."; 
	  return errorChar;}

        if ( result == cudaErrorInvalidValue ) {  errorChar =  "This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.";
	  return errorChar;}

        if ( result == cudaErrorInvalidPitchValue ) {  errorChar =  "This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch.";
	  return errorChar; }

        if ( result == cudaErrorInvalidSymbol ) {  errorChar =  "This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.";
	  return errorChar; }

        if ( result == cudaErrorMapBufferObjectFailed ) {  errorChar =  "This indicates that the buffer object could not be mapped."; 
	  return errorChar;}

        if ( result == cudaErrorUnmapBufferObjectFailed ) {  errorChar =  "This indicates that the buffer object could not be unmapped."; 
	  return errorChar;}

        if ( result == cudaErrorInvalidHostPointer ) {  errorChar =  "This indicates that at least one host pointer passed to the API call is not a valid host pointer."; 
	  return errorChar;}

        if ( result == cudaErrorInvalidDevicePointer ) {  errorChar =  "This indicates that at least one device pointer passed to the API call is not a valid device pointer.";
	  return errorChar;}

        if ( result == cudaErrorInvalidTexture ) {  errorChar =  "This indicates that the texture passed to the API call is not a valid texture."; 
	  return errorChar;}

        if ( result == cudaErrorInvalidTextureBinding ) {  errorChar =  "This indicates that the texture binding is not valid. This occurs if you call cudaGetTextureAlignmentOffset() with an unbound texture."; 
	  return errorChar;}

        if ( result == cudaErrorInvalidChannelDescriptor ) {  errorChar =  "This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by cudaChannelFormatKind, or if one of the dimensions is invalid."; 
	  return errorChar;}

        if ( result == cudaErrorInvalidMemcpyDirection ) {  errorChar =  "This indicates that the direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind."; 
	  return errorChar;}

        if ( result == cudaErrorAddressOfConstant ) {  errorChar =  "This indicated that the user has taken the address of a constant variable, which was forbidden up until the CUDA 3.1 release.";
	  return errorChar; }

        if ( result == cudaErrorTextureFetchFailed ) {  errorChar =  "This indicated that a texture fetch was not able to be performed. This was previously used for device emulation of texture operations."; 
	  return errorChar;}

        if ( result == cudaErrorTextureNotBound ) {  errorChar =  "This indicated that a texture was not bound for access. This was previously used for device emulation of texture operations."; 
	  return errorChar;}

        if ( result == cudaErrorSynchronizationError ) {  errorChar =  "This indicated that a synchronization operation had failed. This was previously used for some device emulation functions."; 
	  return errorChar;}

        if ( result == cudaErrorInvalidFilterSetting ) {  errorChar =  "This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA."; 
	  return errorChar;}

        if ( result == cudaErrorInvalidNormSetting ) {  errorChar =  "This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA."; 
	  return errorChar;}

        if ( result == cudaErrorMixedDeviceExecution ) {  errorChar =  "Mixing of device and device emulation code was not allowed.";
	  return errorChar; }

        if ( result == cudaErrorCudartUnloading ) {  errorChar =  "This indicated an issue with calling API functions during the unload process of the CUDA runtime in prior releases."; 
	  return errorChar;}

        if ( result == cudaErrorUnknown ) {  errorChar =  "This indicates that an unknown internal error has occurred."; 
	  return errorChar;}

        if ( result == cudaErrorNotYetImplemented ) {  errorChar =  "This indicates that the API call is not yet implemented. Production releases of CUDA will never return this error."; 
	  return errorChar;}

        if ( result == cudaErrorMemoryValueTooLarge ) {  errorChar =  "This indicated that an emulated device pointer exceeded the 32-bit address range."; 
	  return errorChar;}

        if ( result == cudaErrorInvalidResourceHandle ) {  errorChar =  "This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like cudaStream_t and cudaEvent_t."; return errorChar;}

        if ( result == cudaErrorNotReady ) {  errorChar =  "This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than cudaSuccess (which indicates completion). Calls that may return this value include cudaEventQuery() and cudaStreamQuery()."; 
	  return errorChar;}

        if ( result == cudaErrorInsufficientDriver ) {  errorChar =  "This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run."; return errorChar;}

        if ( result == cudaErrorSetOnActiveProcess ) {  errorChar =  "This indicates that the user has called cudaSetDevice(), cudaSetValidDevices(), cudaSetDeviceFlags(), cudaD3D9SetDirect3DDevice(), cudaD3D10SetDirect3DDevice, cudaD3D11SetDirect3DDevice(), * or cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by calling non-device management operations (allocating memory and launching kernels are examples of non-device management operations). This error can also be returned if using runtime/driver interoperability and there is an existing CUcontext active on the host thread.";
	  return errorChar; }

        if ( result == cudaErrorInvalidSurface ) {  errorChar =  "This indicates that the surface passed to the API call is not a valid surface."; 
	  return errorChar;}

        if ( result == cudaErrorNoDevice ) {  errorChar =  "This indicates that no CUDA-capable devices were detected by the installed CUDA driver."; 
	  return errorChar;}

        if ( result == cudaErrorECCUncorrectable ) {  errorChar =  "This indicates that an uncorrectable ECC error was detected during execution."; 
	  return errorChar;}

        if ( result == cudaErrorSharedObjectSymbolNotFound ) {  errorChar =  "This indicates that a link to a shared object failed to resolve."; 
	  return errorChar;}

        if ( result == cudaErrorSharedObjectInitFailed ) {  errorChar =  "This indicates that initialization of a shared object failed.";
	  return errorChar; }

        if ( result == cudaErrorUnsupportedLimit ) {  errorChar =  "This indicates that the cudaLimit passed to the API call is not supported by the active device."; 
	  return errorChar;}

        if ( result == cudaErrorDuplicateVariableName ) {  errorChar =  "This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name."; return errorChar;}

        if ( result == cudaErrorDuplicateTextureName ) {  errorChar =  "This indicates that multiple textures (across separate CUDA source files in the application) share the same string name.";
	  return errorChar;}

        if ( result == cudaErrorDuplicateSurfaceName ) {  errorChar =  "This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name.";
	  return errorChar; }

        if ( result == cudaErrorDevicesUnavailable ) {  errorChar =  "This indicates that all CUDA devices are busy or unavailable at the current time. Devices are often busy/unavailable due to use of cudaComputeModeExclusive, cudaComputeModeProhibited or when long running CUDA kernels have filled up the GPU and are blocking new work from starting. They can also be unavailable due to memory constraints on a device that already has active CUDA work being performed.";
	  return errorChar; }

        if ( result == cudaErrorInvalidKernelImage ) {  errorChar =  "This indicates that the device kernel image is invalid.";
	  return errorChar; }

        if ( result == cudaErrorNoKernelImageForDevice ) {  errorChar =  "This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration."; 
	  return errorChar;}

        if ( result == cudaErrorIncompatibleDriverContext ) {  errorChar =  "This indicates that the current context is not compatible with this the CUDA Runtime. This can only occur if you are using CUDA Runtime/Driver interoperability and have created an existing Driver context using the driver API. The Driver context may be incompatible either because the Driver context was created using an older version of the API, because the Runtime API call expects a primary driver contextand the Driver context is not primary, or because the Driver context has been destroyed. Please see Interactions with the CUDA Driver API for more information."; return errorChar;}

        if ( result == cudaErrorPeerAccessAlreadyEnabled ) {  errorChar =  "This error indicates that a call to cudaDeviceEnablePeerAccess() is trying to re - enable peer addressing on from a context which has already had peer addressing enabled.";
	  return errorChar; }

        if ( result == cudaErrorPeerAccessNotEnabled ) {  errorChar =  "This error indicates that cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via cudaDeviceEnablePeerAccess().";
	  return errorChar; }

        if ( result == cudaErrorDeviceAlreadyInUse ) {  errorChar =  "This indicates that a call tried to access an exclusive - thread device that is already in use by a different thread."; 
	  return errorChar;}

        if ( result == cudaErrorProfilerDisabled ) {  errorChar =  "This indicates profiler has been disabled for this run and thus runtime APIs cannot be used to profile subsets of the program. This can happen when the application is running with external profiling tools like visual profiler.";
	  return errorChar; }

        if ( result == cudaErrorProfilerNotInitialized ) {  errorChar =  "This indicates profiler has not been initialized yet. cudaProfilerInitialize() must be called before calling cudaProfilerStart and cudaProfilerStop to initialize profiler.";
	  return errorChar;}

        if ( result == cudaErrorProfilerAlreadyStarted ) {  errorChar =  "This indicates profiler is already started. This error can be returned if cudaProfilerStart() is called multiple times without subsequent call to cudaProfilerStop().";
	  return errorChar;}

        if ( result == cudaErrorProfilerAlreadyStopped ) {  errorChar =  "This indicates profiler is already stopped. This error can be returned if cudaProfilerStop() is called without starting profiler using cudaProfilerStart().";
	  return errorChar; }

        if ( result == cudaErrorStartupFailure ) {  errorChar =  "This indicates an internal startup failure in the CUDA runtime."; 
	  return errorChar;}

        if ( result == cudaErrorApiFailureBase ) {  errorChar =  "Any unhandled CUDA driver error is added to this value and returned via the runtime. Production releases of CUDA should not return such errors. ";
	  return errorChar; }

        return unkonwnerror;
    }
};//struct cuda_result_assert 

#define cuda_assert(result) cuda_result_assert()(result, __FILE__, __LINE__)

#endif//NDEBUG

#endif//_CUDA_ASSERT_HPP_INCLUDED_OPINAFSDLKJNSAFLKJNSADFLJNSAFDLKJNSFDLKJANSFFD 