cmake_minimum_required(VERSION 3.8)
project(task4b)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python executables
install(PROGRAMS
  scripts/task4b_manipulation.py
  scripts/task4b_perception.py
  scripts/shape_detector_task4b.py
  scripts/ebot_nav_task4b.py
  scripts/task4b_navigation.py
  scripts/task4b_shape_detector.py
  scripts/task4b_message_publisher.py
  scripts/simple_waypoint_test.py
  scripts/shape_detection_test.py
  scripts/cartesian_velocity_control.py
  scripts/joint_jog_control.py
  scripts/magnet.py
  scripts/tcp_pose_printer.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
