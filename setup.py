import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NLoed", # Replace with your own username
    version="0.0.1",
    author="Nate Braniff",
    description="A package for nonlinear optimal experimental design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NateBraniff/NLoed",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)