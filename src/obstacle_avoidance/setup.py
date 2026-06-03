from setuptools import find_packages, setup

package_name = 'obstacle_avoidance_pb'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'avoider_node=obstacle_avoidance_pb.avoider_node:main',
            'bug0=obstacle_avoidance_pb.bug0:main',
            'bug1=obstacle_avoidance_pb.bug1:main',
            'bug2=obstacle_avoidance_pb.bug2:main',
            

        ],
    },
)
