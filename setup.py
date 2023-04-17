from setuptools import setup, find_packages

with open("requirements.txt") as fh:
    requirements = [l.strip() for l in fh]


setup(
    name="entqa",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
)
