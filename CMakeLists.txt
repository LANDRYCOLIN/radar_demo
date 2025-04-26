cmake_minimum_required(VERSION 3.8)
project(video_detection_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# 包括目录
include_directories(include)

# 先定义可执行文件
add_executable(video_detection_node 
  src/main.cpp
  src/video_detection_node.cpp
)

# 然后才能使用target_link_libraries
target_link_libraries(video_detection_node
  ${OpenCV_LIBRARIES}
  ${pybind11_LIBRARIES}
  Python3::Python
)

# 其他target_*命令
target_include_directories(video_detection_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OpenCV_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
)

ament_target_dependencies(video_detection_node
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)

# 安装目标和模型文件
install(TARGETS video_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

# 确保模型被安装到正确位置
install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

install(FILES 
  models/best.pt
  DESTINATION share/${PROJECT_NAME}/models
)

# 测试配置（可选）
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
