from setuptools import setup
import os
from glob import glob

package_name = 'puzzle_sim_pb'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*.stl')),
        (os.path.join('share', package_name, 'xacro'), glob('xacro/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Felipe',
    maintainer_email='todo@todo.com',
    description='PuzzleBot simulation',
    license='MIT',
    entry_points={'console_scripts': []},
)