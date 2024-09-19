"""
Setup module for ADP3D.
"""

import codecs
import os
import time

from setuptools import find_packages, setup

with open("requirements.txt", "r") as req_file:
    requirements = [line.split("#")[0].strip() for line in req_file]
    requirements = [line for line in requirements if line]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


version = get_version("adp3d/__init__.py")

# During CICD, append "-dev" and unix timestamp to version
if os.environ.get("CI_COMMIT_BRANCH") == "develop":
    version += f".dev{int(time.time())}"

setup(
    name="adp3d",
    version=version,
    url="https://github.com/k-chrispens/adp3d",
    packages=find_packages(),
    description="Tool to sample protein conformations into density",
    include_package_data=True,
    author="Karson Chrispens",
    license="MIT",
    install_requires=requirements,
)