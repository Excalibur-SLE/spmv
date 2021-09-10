add_executable(test_spmv test_spmv.cpp)
target_link_libraries(test_spmv LINK_PUBLIC ${PROJECT_NAME})

add_test(spmv_n1 ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_spmv)
add_test(spmv_n2 ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_spmv)

if(OpenMP_CXX_FOUND)
  set_tests_properties(spmv_n1 spmv_n2 PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
endif()
if(USE_DPCPP)
  set_tests_properties(spmv_n1 spmv_n2 PROPERTIES ENVIRONMENT "DPCPP_CPU_NUM_CUS=2")
endif()
