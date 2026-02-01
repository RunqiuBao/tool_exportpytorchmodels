cmake_minimum_required(VERSION 3.14)
project(pytorch_custom_ops LANGUAGES CXX CUDA)

# Ask PyTorch which ABI it was built with (0 or 1)
execute_process(
    COMMAND python3 -c "import torch; print(int(torch.compiled_with_cxx11_abi()))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
message(STATUS "PyTorch ABI version: ${TORCH_CXX11_ABI} (0=Old, 1=New)")

# --- Find CUDA ---
set(CUDA_LIB_DIR "/usr/local/cuda/targets/x86_64-linux/lib")
set(CUDA_INCLUDE_DIR "/usr/local/cuda/targets/x86_64-linux/include")

# --- Find TensorRT ---
set(TENSORRT_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "Path to TensorRT root")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR} PATH_SUFFIXES include
)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_DIR} PATH_SUFFIXES lib lib64
)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_DIR.")
endif()
# --- Find PyTorch ---
execute_process(COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE TORCH_CMAKE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)

# --- Include Header Directories ---
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mmdeploy/csrc
    ${CUDA_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIRS}
)

# --- Build TensorRT Plugin Library ---
# Source files typically found in: 
# csrc/mmdeploy/backend_ops/tensorrt/modulated_deform_conv/
add_library(mmdeploy_trt_ops SHARED
    3rdparty/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/modulated_deform_conv/trt_modulated_deform_conv.cpp
    3rdparty/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/modulated_deform_conv/trt_modulated_deform_conv_kernel.cu
    3rdparty/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/common_impl/trt_cuda_helper.cu
)
target_include_directories(mmdeploy_trt_ops PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/common
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mmdeploy/csrc/mmdeploy/backend_ops/common/
)
target_link_libraries(mmdeploy_trt_ops PRIVATE
    -Wl,--no-as-needed
    ${CUDA_LIB_DIR}/libcublas.so
    ${CUDA_LIB_DIR}/libcublasLt.so  # Modern cublas usually needs this too
    ${CUDA_LIB_DIR}/libcudart.so
    -Wl,--as-needed
    nvinfer
    nvonnxparser
)
set_target_properties(mmdeploy_trt_ops PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)

# --- 2. Build Library for registering modulated_deform_conv in pytorch ---
# Compile sources into loose objects (.o files)
#    We do this so we can manually force the linker to accept them all later.
add_library(mmdeploy_torch_ops_objs OBJECT
    # 1. The python binding
    3rdparty/mmdeploy/csrc/mmdeploy/backend_ops/torchscript/ops/bind.cpp
    # 2. The CPU implementation
    3rdparty/mmdeploy/csrc/mmdeploy/backend_ops/torchscript/ops/modulated_deform_conv/modulated_deform_conv_cpu.cpp
    # 3. The CUDA implementation
    3rdparty/mmdeploy/csrc/mmdeploy/backend_ops/torchscript/ops/modulated_deform_conv/modulated_deform_conv_cuda.cu
)
target_include_directories(mmdeploy_torch_ops_objs PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mmdeploy/csrc/mmdeploy/backend_ops/common/
    ${CUDA_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
)
set_target_properties(mmdeploy_torch_ops_objs PROPERTIES 
    CXX_STANDARD 17 
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)
# add_library needs at least one source file.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp "")
# Create the Shared Library using the dummy file
add_library(mmdeploy_torch_ops SHARED ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)
target_link_libraries(mmdeploy_torch_ops PRIVATE
    -Wl,--whole-archive
    $<TARGET_OBJECTS:mmdeploy_torch_ops_objs>
    -Wl,--no-whole-archive
    ${TORCH_LIBRARIES}
    ${CUDA_LIB_DIR}/libcudart.so
)
execute_process(COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')"
                OUTPUT_VARIABLE TORCH_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
message("baodebug: TORCH_INSTALL_PATH: ${TORCH_LIB_DIR}")
set_target_properties(mmdeploy_torch_ops PROPERTIES
    PREFIX "lib"
    SUFFIX ".so"
    # Ensure symbols are visible
    CXX_VISIBILITY_PRESET default
    # Handle the RPATH so it finds libtorch.so
    INSTALL_RPATH "${TORCH_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)


# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 80)


# Define the installation targets
install(TARGETS mmdeploy_trt_ops mmdeploy_torch_ops
    LIBRARY DESTINATION lib    # Where .so files go
    ARCHIVE DESTINATION lib    # Where .a files go (if any)
    RUNTIME DESTINATION bin    # Where .exe or .dll go (if on Windows)
)
# Install the headers
install(DIRECTORY 3rdparty/mmdeploy/csrc/ DESTINATION include/mmdeploy)