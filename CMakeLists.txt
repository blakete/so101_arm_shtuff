cmake_minimum_required(VERSION 3.16)
project(so101_arm_shtuff CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# librealsense2 here is a source install under /usr/local, so go through
# pkg-config rather than a config package that may not be exported.
find_package(PkgConfig REQUIRED)
pkg_check_modules(REALSENSE2 REQUIRED IMPORTED_TARGET realsense2)
pkg_check_modules(OPENCV4 REQUIRED IMPORTED_TARGET opencv4)

# Optional: used only to query the screen size so the window can auto-fit.
pkg_check_modules(X11 IMPORTED_TARGET x11)

add_executable(d455_viewer d455_viewer.cpp)
target_link_libraries(d455_viewer PRIVATE PkgConfig::REALSENSE2 PkgConfig::OPENCV4)
if(X11_FOUND)
  target_link_libraries(d455_viewer PRIVATE PkgConfig::X11)
  target_compile_definitions(d455_viewer PRIVATE HAVE_X11)
else()
  message(STATUS "libx11 not found - window will assume a 1920x1080 screen")
endif()

# /usr/local/lib is where librealsense2.so lives; bake it in so the binary
# runs without LD_LIBRARY_PATH.
set_target_properties(d455_viewer PROPERTIES
  BUILD_RPATH "/usr/local/lib"
  INSTALL_RPATH "/usr/local/lib")
