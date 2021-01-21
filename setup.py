#from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nloed", 
    version="0.0.2",
    author='Nate Braniff',
    author_email='nbraniff@uwaterloo.ca',
    description="A package for nonlinear optimal experimental design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NateBraniff/NLoed",
    keywords = ['Optimal','Experimental','Design'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6.1',
    install_requires = [
        'casadi>=3.5.0',
        'numpy>=1.17.0',
        'pandas>=1.0.0',
        'scipy',
        'matplotlib',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license = 'LGPLv3+'
    # py_modules=["model","design"],
    # package_dir={'':'nloed'},
)