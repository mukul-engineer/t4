#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Navigation node
        Node(
            package='task4b',
            executable='ebot_nav_task4b.py',
            name='ebot_navigation',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ]
        ),
        
        # Perception node
        Node(
            package='task4b',
            executable='task4b_perception.py',
            name='fruit_pose_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ]
        ),
        
        # Shape detection node
        Node(
            package='task4b',
            executable='shape_detector_task4b.py',
            name='lidar_line_ransac',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ]
        ),
        
        # Manipulation node
        Node(
            package='task4b',
            executable='task4b_manipulation.py',
            name='hybrid_waypoint_servo_node_v7',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ]
        ),
    ])
