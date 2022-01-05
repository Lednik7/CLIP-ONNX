import os
import pkg_resources
from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    install_requires = f.read().split("\n")

install_requires += ["clip @ git+https://github.com/openai/CLIP.git"]

setup(
    name="clip_onnx",
    version="1.0",
    description="",
    author="Maxim Gerasimov",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True
)