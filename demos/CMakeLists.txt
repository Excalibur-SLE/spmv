
# Executables

add_executable(cg cgmain.cpp)
target_link_libraries(cg LINK_PUBLIC ${PROJECT_NAME})

if(USE_SYCL)
  add_executable(cg_sycl cgmain_sycl.cpp)
  target_link_libraries(cg_sycl LINK_PUBLIC ${PROJECT_NAME})
endif()

add_executable(matvecmain matvecmain.cpp CreateA.cpp)
target_link_libraries(matvecmain ${PROJECT_NAME})

add_executable(restrictmain restrictmain.cpp)
target_link_libraries(restrictmain ${PROJECT_NAME})
