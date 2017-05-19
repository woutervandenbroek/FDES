#############################
# Check for GPUs  and their compute capability
#modified url: https://github.com/jwetzl/CudaLBFGS/blob/master/CheckComputeCapability.cmake

if(CUDA_FOUND)
message(STATUS "${CMAKE_MODULE_PATH}cuda_compute_capability.c") 
try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
${CMAKE_MODULE_PATH}
${CMAKE_MODULE_PATH}/cuda_compute_capability.c
CMAKE_FLAGS
-DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
-DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)
message(STATUS "${RUN_OUTPUT_VAR}")
set(CUDA_CAPABILITY_INFO ${RUN_OUTPUT_VAR} CACHE STRING "Compute capability of CUDA-capable GPU list")
endif()

