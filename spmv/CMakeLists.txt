set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
add_library(${PROJECT_NAME} SHARED L2GMap.cpp Matrix.cpp read_petsc.cpp cg.cpp)

target_include_directories (${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (${PROJECT_NAME} INTERFACE $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen MPI::MPI_CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
if(BLAS_FOUND)
  target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_USE_MKL_ALL)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${BLAS_LIBRARIES})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "spmv.h")
set_target_properties(${PROJECT_NAME} PROPERTIES PRIVATE_HEADER "L2GMap.h;Matrix.h;read_petsc.h;cg.h")

install(TARGETS ${PROJECT_NAME}
  EXPORT libspmv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include
  PRIVATE_HEADER DESTINATION include
  )

# First attempt at installing a config
include(CMakePackageConfigHelpers)
write_basic_package_version_file( 
  spmvConfig.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion
  )

install(EXPORT libspmv
  FILE spmvConfig.cmake
  NAMESPACE spmv::
  DESTINATION share/spmv/cmake)
