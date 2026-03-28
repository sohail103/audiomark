include_directories(
    ${PORT_DIR}
    ${PORT_DIR}/src
)

file(GLOB PORT_SOURCES
    ${PORT_DIR}/th_api.c
    ${PORT_DIR}/src/*.c
    ${PORT_DIR}/src/p_riscv_cfft/*.c
    ${PORT_DIR}/src/p_riscv_conv/*.c
)

# TODO: Add our FFT functions as an FFT backend in fftwrap.c
# for now, we enable but do not use smallfft since fftwrap.c
# throws an error without it.
add_definitions(-DUSE_SMALLFT)

set(PORT_SOURCE 
    ${PORT_SOURCES}
)
