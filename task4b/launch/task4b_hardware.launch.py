#!/usr/bin/env python3
"""
Task 4B Hardware Launch File
Launches: navigation, shape_detector, message_publisher
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Message Publisher Node
        Node(
            package='task4b',
            executable='task4b_message_publisher.py',
            name='task4b_message_publisher',
            output='screen',
        ),
        
        # Shape Detector Node
        Node(
            package='task4b',
            executable='task4b_shape_detector.py',
            name='task4b_shape_detector',
            output='screen',
        ),
        
        # Navigation Node (starts last)
        Node(
            package='task4b',
            executable='task4b_navigation.py',
            name='task4b_navigation',
            output='screen',
        ),
    ])
