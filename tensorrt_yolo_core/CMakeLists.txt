cmake_minimum_required(VERSION 3.5)
project(tensorrt_yolo_ros2)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 添加 include 目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 查找并设置 ROS2 依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tensorrt_yolo_msg REQUIRED)
find_package(geometry_msgs REQUIRED)
# 查找并设置 PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
# 查找 CUDA
# ============= cuda ============
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# 查找 TensorRT
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("Embed_platform on")
  include_directories(/usr/include/aarch64-linux-gnu)
  link_directories(/usr/lib/aarch64-linux-gnu)
else()
  message("Embed_platform off")
  include_directories(/usr/include/x86_64-linux-gnu)
  link_directories(/usr/lib/x86_64-linux-gnu)
endif()

# 查找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置 CUDA 文件
cuda_add_library(yolo_infer
        src/infer.cpp
        src/calibrator.cpp
       src/postprocess.cu
        src/preprocess.cu
        src/config.cpp
)
target_link_libraries(yolo_infer
        cudart
        nvinfer
        nvonnxparser
        ${OpenCV_LIBS})
ament_target_dependencies(yolo_infer
        rclcpp
        tensorrt_yolo_msg
        geometry_msgs
        cv_bridge
)
# 添加可执行文件
add_executable(camera_infer_node
  src/camera_infer.cpp
  include/camera_infer.h
  include/byte_tracker.h
  include/bytekalman_filter.h
  include/lapjv.h
  include/strack.h
  src/bytekalman_filter.cpp
  src/byte_tracker.cpp
  src/lapjv.cpp
  src/strack.cpp
)
ament_target_dependencies(camera_infer_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
)

# 链接库
target_link_libraries(camera_infer_node
        yolo_infer
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
)

add_executable(image_infer_node
  src/image_infer.cpp
  include/image_infer.h
)
ament_target_dependencies(image_infer_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
)

target_link_libraries(image_infer_node
  yolo_infer
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)
# 安装目标
install(TARGETS
  camera_infer_node
  image_infer_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
