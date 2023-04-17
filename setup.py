from setuptools import setup, find_packages
from pip.req import parse_requirements

# Parse requirements.txt and extract the dependencies
install_reqs = parse_requirements("requirements.txt", session=False)
requirements = [str(req.req) for req in install_reqs]

setup(
    name="entqa",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
)
