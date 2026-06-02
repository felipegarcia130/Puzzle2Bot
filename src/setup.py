from setuptools import find_packages, setup

package_name = 'mcl_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/maps', [
            'maps/slam_map.npy',
            'maps/slam_log_odds.npy',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='felipe',
    maintainer_email='felipe@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mcl_node = mcl_robot.mcl_node:main',
            'astar_node = mcl_robot.astar_node:main',
            'slam_node =mcl_robot.slam_node:main',
            'exploration_node= mcl_robot.exploration_node:main',
            'odom_node=mcl_robot.odom_node:main',
        ],
    },
)
