target_sources(${LIB_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/openacc_executor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csr_kernels.openacc.cpp
  )

# Extend header path for build and install phases
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/openacc>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "PGI" OR CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
  #-ta=tesla 
  target_compile_options(${LIB_NAME} PRIVATE -Minfo=all -Mneginfo)
endif()

# Install the header files manually, since CMake does not respect subdirectories
set(publicHeaders "openacc_executor.h")
foreach(header ${publicHeaders})
  install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/cuda")
endforeach()
