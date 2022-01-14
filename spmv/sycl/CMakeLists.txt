add_sycl_to_target(TARGET ${LIB_NAME})
target_sources(${LIB_NAME} PRIVATE cg.sycl.cpp)

# Extend header path for build and install phases
target_include_directories(${LIB_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )
target_include_directories(${LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/sycl>
  )

# Set the header files that need to be installed (cannot use property
# in this cuda as it does not preserve directory structure)
set(publicHeaders "cg_sycl.h")
foreach(header ${publicHeaders})
  install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/sycl")
endforeach()
