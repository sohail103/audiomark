include_directories(
    ${PORT_DIR}
    ${PORT_DIR}/muriscv-nn/Include
)

file(GLOB PORT_SOURCES
    ${PORT_DIR}/*.c
    ${PORT_DIR}/src/*.c
    ${PORT_DIR}/src/rvv/*.c
    ${PORT_DIR}/src/rvp/*.c
)

file(GLOB MURISCV_NN_SOURCES
    ${PORT_DIR}/muriscv-nn/Source/ConvolutionFunctions/*.c
    ${PORT_DIR}/muriscv-nn/Source/DepthwiseConvolutionFunctions/*.c
    ${PORT_DIR}/muriscv-nn/Source/FullyConnectedFunctions/*.c
    ${PORT_DIR}/muriscv-nn/Source/PoolingFunctions/*.c
    ${PORT_DIR}/muriscv-nn/Source/SoftmaxFunctions/*.c
    ${PORT_DIR}/muriscv-nn/Source/NNSupportFunctions/*.c
)

# remove unwanted files
list(FILTER MURISCV_NN_SOURCES EXCLUDE REGEX ".*lstm.*")
list(FILTER MURISCV_NN_SOURCES EXCLUDE REGEX ".*svdf.*")
list(FILTER MURISCV_NN_SOURCES EXCLUDE REGEX ".*transpose.*")
list(FILTER MURISCV_NN_SOURCES EXCLUDE REGEX ".*_1_x_n_s4.*")

add_definitions(-DUSE_SMALLFT)

# RISC-V Vector extension support
# Set RISCV_ARCH to your target architecture string
# Examples:
#   rv64gcv        - RV64 with full V extension (includes Zve64f, Zve32f)
#   rv64gc_zve32f  - RV64 with minimal float vector (Zve32f only)
#   rv32gcv        - RV32 with full V extension
#   rv32gc_zve32f  - RV32 with Zve32f
#
# Pass via: -DRISCV_ARCH=rv64gcv_zve32f
if(DEFINED RISCV_ARCH)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=${RISCV_ARCH}")
    message(STATUS "RISC-V architecture: ${RISCV_ARCH}")
endif()

set(PORT_SOURCE 
    ${PORT_SOURCES}
    ${MURISCV_NN_SOURCES}
)
