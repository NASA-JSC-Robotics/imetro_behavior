find_package(ament_cmake_pytest REQUIRED)

ament_add_pytest_test(test_basic_behaviors test_basic_behaviors.py)

ament_add_pytest_test(test_color_behaviors test_color_behaviors.py)

ament_add_pytest_test(test_control_behaviors test_control_behaviors.py)

ament_add_pytest_test(test_decorators test_decorators.py)

ament_add_pytest_test(test_executor test_executor.py)

ament_add_pytest_test(test_geometry_behaviors test_geometry_behaviors.py)

ament_add_pytest_test(test_joint_behaviors test_joint_behaviors.py)

ament_add_pytest_test(test_moveit_behaviors test_moveit_behaviors.py)

ament_add_pytest_test(test_nav_behaviors test_nav_behaviors.py)

ament_add_pytest_test(test_ros_behaviors test_ros_behaviors.py)

ament_add_pytest_test(test_apriltag_behaviors test_apriltag_behaviors.py)

install(
  DIRECTORY test_data/
  DESTINATION share/${PROJECT_NAME}/tests/test_data
)
