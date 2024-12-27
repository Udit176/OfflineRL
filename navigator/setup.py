from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rlplanner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'maps'), glob(os.path.join('maps', '*'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*'))),
        (os.path.join('share', package_name, 'csv'), glob(os.path.join('csv', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nobody',
    maintainer_email='nobody',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gen_sync_map_launch = rlplanner.gen_sync_map_launch:main',
            'navigator_csv = rlplanner.navigator_csv:main',
        ],
    },
)
