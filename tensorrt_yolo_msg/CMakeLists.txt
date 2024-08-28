cmake_minimum_required(VERSION 3.5)
project(tensorrt_yolo_msg)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(Python3 REQUIRED)

set(CMAKE_PREFIX_PATH "/opt/ros/humble/share" ${CMAKE_PREFIX_PATH})
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Results.msg"
  "msg/InferResult.msg"
  "msg/KeyPoint.msg"
 )

# Export ROS2 package dependencies
ament_export_dependencies(rosidl_default_runtime)

# Install interfaces
install(
  DIRECTORY msg
  DESTINATION share/${PROJECT_NAME}
)

# For testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
