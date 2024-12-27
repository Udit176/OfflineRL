
import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration

def generate_launch_description():
   
    
    return LaunchDescription([
        
        # Launch navigator_csv node
        Node(
            package='rlplanner',
            executable='navigator_csv',
            name='navigator_csv',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
    ])
