# NLoed
Nonlinear Optimal Experimental Design
 
## Description
NLoed (suggested pronounciation; "en-load") is an open source Python package for building and 
optimizing experimental designs for non-linear models, with a specific emphasis on applications in systems 
biology. The package is primarily focused on generating optimal designs for improved parameter estimation
using a relaxed formulation of the design optimization problem.

NLoed is built on top of Casadi, an optimal control prototyping framework, and users construct their models using Casadi's symbolic classes. Casadi's symbolic interface allows the user to encode a wide variety of nonlinear mathematical structures into their NLoed model. NLoed supports both static and dynamic nonlinear models including systems of nonlinear differential equations. Casadi also provides a number of numerical
tools such as automatic differentiation and interfaces to nonlinear programming solvers, such as IPOPT,
which greatly improve NLoed's performance. In addition to design, NLoed provides a number of useful model building tools including functions for maximum likelihood fitting, sensitivity analysis, data simulation, asymptotic and profile likelihoods, graphical identifiability diagnostics and quantification of prediction uncertainty. These tools, in conjunction with NLoed's core optimal design functionality, aim to provide a complete model building framework, enabling model construction, experimental design, model fitting and diagnostics to all be done within NLoed. This makes NLoed ideal for model builders heading into the
lab or for those who wish to study model identifiability via simulation studies.

Disclaimer: NLoed is still under development. A reasonable effort has been made to create a stable
functional interface and to test the current release however, users should  proceed with reasonable 
caution in writing code for publication or in developing other numerical tools using NLoed. The interface may be subject to change, especially in advance of the first non-beta (i.e. beta realeases are any that preceed version 1.0). Also, the authours make no guarantee for the absolute exahuastic correctness of all numerical routines implemented in NLoed (or its dependencies), and some reasonable testing of the package with the user's specific model, as well as some understanding of general OED theory are recommended when using the packag (especiall before version 1.0). If you identify any bugs or numerical issues, please submit them
to the authours on Github, your error tickets are greatly appreciated! :) 


## Installation
NLoed is available on PyPI, so you can just use pip to install it!
To install with pip, run:
```sh
pip install nloed
```

Taadaa! You are done! You can then test that it worked by starting up python and importing NLoed with;
```sh
python
[something something something... python starts]
>>>import nloed
[no errors should be thrown]
```
After this it's a good idead to try some of the examples on Github.

### I'm new to python...

If you've never used python or pip before, you may want to check a few things first before trying to
 install NLoed.
First make sure you have python installed, its on your path, and the python version is appropriate.
To do this, run the following on the command line/prompt:
```sh
python -V
```
It should print out a python version number which must be greater than 3.6.1. If this is susccesful,
you should also have pip installed by default (it ships with python >3.4), to confirm this run:
```sh
pip -V
```
This should print out a similar version number as the one python did.

If you run into issues with finding python on your system, you will need to do some
trouble shooting. For those unfamiliar with python, here are some helpful pointers depending on 
your OS:

#### OSX
Macs ship with Python 2.7 by default which doesn't normally come with pip installed and isn't
compatible with NLoed anyways. 

There are two simple options for getting Python >3.6.1 on your Mac:

##### Use Homebrew (https://brew.sh/)
This is useful if you aren't doing much with Python other than NLoed, but it is a bit less flexible.

* Make sure Xcode dependencies are installed (Homebrew needs these to do stuff):
```sh
xcode-select --install
```

* Install Homebrew, its a package manager for installing software on Macs:
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

* Check Homebrew is installed properly:
```sh
brew --version
[should print out a version number]
```

* Install Python 3 with:
```sh
brew install python3
```

* Check python 3 is installed with:
```sh
python3 -V
```

* As the command 'python' points to the native 2.7 installed by default on your Mac, you will
now need to use 'python3' to start the newer version installed with Homebrew. To install NLoed on
your python 3 installation specifically, run:
```sh
$ pip3 install nloed
```
Here Homebrew has setup 'pip3' to replace 'pip' in order to avoid confusion.

##### 2) Use Pyenv (https://github.com/pyenv/pyenv)
This method is better if you are going to do a lot of work in Python and want to organize many 
versions and dependencies.

* Make sure Homebrew is installed as above.

* Run the following to install some needed dependencies;
```sh
brew install openssl readline sqlite3 xz zlib
```

* Install pyenv either with
a) Homebrew itself:
```sh
brew install pyenv
```
b) The auto-installer script (https://github.com/pyenv/pyenv-installer):
```sh
curl https://pyenv.run | bash
```

* Run the following to set up your shell environment:
```sh
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
exec "$SHELL"
```

* Check pyenv is installed correctly:
```sh
pyenv versions
[prints out some version, likely just 'system']
```

* Install a version of python greater than 3.6.1, for example:
```sh
pyenv install 3.7.0
```

* You then need to make this new version active, the easiest way to do it is to make it the global
default by running:
```sh
pyenv global 3.7.0
```
Note the version number here should be the same one you installed in the previous step.
Check you the pyenv documentation for more options for managing versions, it is very flexible.

* Finally install the NLoed package on the new pyenv version by running:
```sh
pip install nloed
```

#### Linux (Ubuntu/Debian)

#### Windows

# Dependencies & Versions
NLoed requires Python 3.6.1 or newer, Python 2 is not supported.

NLoed uses the following (direct) dipendencies and has been tested with the versions indicated below:
* Casadi>=3.5.0
* Numpy>=1.17.0
* Pandas>=1.0.0
* Scipy
* Matplotlib

# Usage
NLoed consistes of two core class: the Model class and the Design class. The Model class is used to 
encode a specific mathematical model as well as to perform basic model analysis including;
fitting, sensitivity and uncertainty analysis, confidence intervals, fitting diagnostics, data simulation
and generating model predictions. The NLoed Model class supports multi-input, multi-output models
with a variety possible error distributions available for model observations. This makes NLoed suitable
for modelling process that generate binary and count data as well as other skewed, non-normal distributions
The Model class supports hetroskedastic models, including those with variance specific parameters. 
At this time the Model class does not support models with correlated observations and all observations
in NLoed are assumed to be independent. The Design class in NLoed is used to generate optimal designs
meeting specification and constratins specified by the user. User's pass their Model class instance
into the Design class to generate an archetypal (relaxed) design. The Design class then provides the
user with functionality to generate a variety of implementable (exact) designs with a desired sample
size. The majority of input and output data for NLoed is handled via Pandas dataframes, which makes
it easy to import data and export designs or simulations to third-party software. 

NLoed's basic workflow involves construcing Model class instances to encode a user's mathematical model
and then using the Model instance to create a Design instance encoding an experimental design for a 
desired scenario. Use of the Model and Design class can be done iterativly. The Design class can be 
used to generate experiments and the Model class can then be used to fit the resulting data. Model refinments  depending on the 
users desired goals.

Further examples can be found in the \example folder on NLoed's Github. A full description of the OED background, NLoed's use cases and function call formats can be found in the \docs folder on Github.

# Contributing
NLoed is maintained by the Ingalls Lab, in the Department of Applied Math, at the University of Waterloo.

Inquiries about contributions and usage can be directed to Brian Ingalls at bingalls[at]uwaterloo.ca 

Issue ticket submission and general discussion can be done through Github.

# Credits
NLoed was written by Nathan Braniff and Brian Ingalls.

Special thanks to:
* Michael Astwood for early prototyping of the project
* Zixuan Liu and Taylor Pearce for testing some experimental designs generated by NLoed in the wetlab

# License

