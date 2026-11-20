cmake_minimum_required(VERSION 3.22)
project(imetro_behavior_testing)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/run_behavior_trials.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

ament_package()
