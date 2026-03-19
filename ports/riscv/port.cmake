include_directories(
    ${PORT_DIR}
    ${PORT_DIR}/muriscv-nn/Include)

file(GLOB PORT_SOURCES
    ${PORT_DIR}/*.c
    ${PORT_DIR}/scalar/*.c
    ${PORT_DIR}/vector/*.c
    ${PORT_DIR}/packed_simd/*.c
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

set(PORT_SOURCE 
    ${PORT_SOURCES}
    ${MURISCV_NN_SOURCES}
)
