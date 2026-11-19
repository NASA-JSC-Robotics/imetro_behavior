find_package(ament_cmake_pytest REQUIRED)

ament_add_pytest_test(test_trials test_trials.py)

ament_add_pytest_test(test_runner test_runner.py)
