from setuptools import find_packages, setup


setup(
    name="trajectory-transformer",
    version="0.1.0",
    packages=find_packages(include=["trajectory", "trajectory.*"]),
    include_package_data=True,
)
