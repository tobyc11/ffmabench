cmake_minimum_required(VERSION 3.16)

if(APPLE)
    project(ffmabench OBJC OBJCXX)

    add_executable(ffmabenchmetal ffmabenchmetal.m)
    target_link_libraries(ffmabenchmetal "-framework CoreFoundation")
    target_link_libraries(ffmabenchmetal "-framework Metal")
    add_custom_command(
        TARGET ffmabenchmetal POST_BUILD
        COMMAND xcrun metal ${CMAKE_SOURCE_DIR}/metalgpucode.metal
        VERBATIM
    )
else()
    project(ffmabench CXX CUDA)

    add_executable(ffmabench ffmabench.cu)
    set_property(TARGET ffmabench PROPERTY CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86)
endif()
