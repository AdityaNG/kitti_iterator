from setuptools import setup
from setuptools import find_packages


def load(path):
    return open(path, 'r').read()

kitti_iterator_version = '1.0.3'

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering"
]


if __name__ == "__main__":
    setup(
        name="KITTI Iterator",
        version=kitti_iterator_version,
        description="KITTI Iterator",
        long_description=load('README.md'),
        long_description_content_type='text/markdown',
        platforms="OS Independent",
        package_data={'kitti_iterator': ['README.md']},
        packages=find_packages(exclude=['tests']),
        install_requires=["pandas"],
    )
