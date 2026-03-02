from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pf_localization',
            executable='particle_filter_node',
            name='pf_node',
            output='screen',
            parameters=[
                {'num_particles': 300}
            ]
        )
    ])
