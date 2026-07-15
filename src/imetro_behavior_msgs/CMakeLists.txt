cmake_minimum_required(VERSION 3.22)
project(imetro_behavior_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(moveit_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  action/ExecuteBehavior.action
  action/PreviewTrajectory.action
  DEPENDENCIES moveit_msgs
)

ament_package()
