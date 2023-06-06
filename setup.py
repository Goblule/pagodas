from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='pagodas',
      version="1.0",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests")
