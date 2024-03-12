from setuptools import find_packages, setup

package_name = 'simple_robot_controller'

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
    maintainer='luke',
    maintainer_email='lmnewey@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_robot_controller = simple_robot_controller.simple_robot_controller:main',
            'pidtune = pidtune.pidtune:main',
            'waypointpublisher = waypointpublisher.waypointpublisher:main'
        ],
    },
)
