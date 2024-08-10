from setuptools import find_packages, setup

short_description = (
    "Use typer cli and dataclass together to manager commandline args and your settings"
)
with open("README.md", "r") as readme:
    long_description = readme.read()
packages = find_packages(exclude=["tests*"])
requires = []

setup(
    name="dataclass_runner",
    version="0.0.1",
    packages=packages,
    license="MIT",
    description=short_description,
    long_description=long_description,
    install_requires=requires,
    author="Blender Wang",
    author_email="developinblend@gmail.com",
)
