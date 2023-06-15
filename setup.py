from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='pagodas',
      version="1.0",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      license="MIT",
      url="https://github.com/Goblule/pagodas",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
