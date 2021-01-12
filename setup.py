#from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nloed", 
    version="0.0.1",
    author='Nate Braniff',
    author_email='nbraniff@uwaterloo.ca',
    description="A library for nonlinear optimal experimental design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NateBraniff/NLoed",
    keywords = ['Optimal','Experimental','Design'],
    packages=setuptools.find_packages(),
    install_requires = [
        'casadi',
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    # py_modules=["model","design"],
    # package_dir={'':'nloed'},
)