# NLoed Development Guide

This document summarizes the environment setup and some of the installation steps needed to contribute
to development of the NLoed package. This guide is for UNIX-based operating systems, specifically
OSX or Ubuntu-like Linux distrobutions.

If you have a Windows machine you can attempt to follow this guide very roughly using tools like [pyenv-win](https://github.com/pyenv-win/pyenv-win), however it may be easier to install something like VirtualBox and the latest Ubuntu release and do the development on a virtual machine.

* **[The Development Environment](#the-development-environment)**
* **[Testing with Pytest and Tox](#testing-with-pytest-and-tox)**
* **[Latex Documentation with Sphinx and Overleaf](#latex-documentation-with-sphinx-and-overleaf)**
* **[Notebook Examples and the Example Pack](#notebook-examples-and-the-example-pack)**
* **[Packaging and Release](#packaging-and-release)**

## The Development Environment

This section explains how to set up a virtual environment, git clone the repo, install development
dependencies and do a local install of the package for development.

1. **Install Pyenv and Pyenv-virtualenv**
    It's recommended that you use a virtual environment for development to keep dependencies and versions
    well organized. There are a few options here, in this guide we focus on using [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv):

    *OSX*
    * Install Xcode dependencies:
        ```sh
        xcode-select --install
        ```
    * Install Homebrew:
        ```sh
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        ```
    *  Install Python build dependencies:
        ```sh
        brew install openssl readline sqlite3 xz zlib
        ```
    * Install Pyenv:
        with Homebrew (needs update of .bashrc and environment variables, also a shell restart),
        ```sh
        brew install pyenv
        echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
        exec "$SHELL"
        ```
        or with auto-install script (needs shell restart),
        ```sh
        curl https://pyenv.run | bash
        exec "$SHELL"
        ```
    * Install pyenv-virtualenv:
        ```sh
        brew install pyenv-virtualenv
        ```

    *Linux (Ubuntu/Debian)*
    * Install Python build dependencies:
        ```sh
        sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
        ```
    * Install pyenv using auto-install script and restart the shell:
        ```sh
        curl https://pyenv.run | bash
        ```
    * Install pyenv-virtualenv (see next steps for Git installation):
        ```sh
        git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
        echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
        exec "$SHELL
        ```

2. **Clone Git Repo into the Development Folder**
    You need to create a folder to contain the package files. You can do this in the file manager or
    from the commandline, after which Git is used to clone the NLoed repository.
    * Open Terminal
    * Navigate to where you want the development folder:
        ```sh
        cd ~/path/to/where/dev/folder/goes/
        ```
    * Create the folder:
        ```sh
        mkdir name_of_dev_folder
        ```
    * Enter development folder:
        ```sh
        cd name_of_dev_folder
        ```
    * Install Git:
        *OSX*
        ```sh
        brew install git
        ```
        *Linux (Ubuntu/Debian)
        ```sh
        sudo apt-get install git
        ```
    * Clone the git repo (run in dev directory):
        ```sh
        git clone https://github.com/NateBraniff/NLoed
        ```

3. **Install Python and Dependencies in a Virtual Environment**
    Using Pyenv we now install whichever version of Python we wish to use for development. NLoed was
    originally developed using Python 3.7.6 and this version is used below but you should upgrade 
    appropriatly as new Python versions are released.

    * Install Python 3.7.6
        ```sh
        pyenv install -v 3.7.6
        ```
    * Create a Virtual Environment
        ```sh
        pyenv virtualenv 3.7.6 name_of_virtual_environment
        ```
    * Set the Virtual Environment as Local in the Development Directory (run in dev directory)
        ```sh
        pyenv local name_of_virtual_environment
        ```
    * Install Development Dependencies (run in dev directory):
        ```sh
        pip install -r requirements.txt
        ```
    * Install NLoed Locally in Edit Mode (run in dev directory):
        ```sh
        pip install -e
        ```

## Testing with Pytest and Tox
NLoed has been set up to use [Pytest](https://docs.pytest.org/en/stable/) for testing the package's classes and functions.
[Tox](https://tox.readthedocs.io/en/latest/) is used to run the Pytest tests with various python versions and dependencies.
These packages should be installed during the steps outlined above the the requirments.txt file.

1. **Running Pytest**
    Pytest will run every test function in the files prexided with 'test' in the test folder. It will
    confirm the assert statments at the end of the test function are true and generate a report upon 
    completion indicating which tests failed.
    * Run Pytest (from dev directory):
        ```sh
        pytest
        ```

2. **Running Tox**
    Tox extends the functionality of Pytest by creating a series of virtual environments and installing 
    the package and various python version and dependency combinations in each. In each virtual environment
    it then runs the Pytest tests to confirm that various version and dependency combinations are supported.
    The exact combination of versions and dependencies is specified in the tox.ini 
    * Run Tox (from dev directory):
        ```sh
        pytest
        ```

3. **Running Tox on OS/Linux/Windows**
    It may be a good idea to test new releases using tox on various platforms.  Most of the code in NLoed -- in its initial release -- is platform independent but it is probably a good idea to continue to test from time to
    time. This is somewhat complex and depends on what OS's and machines you have available, a rough
    guide is given below:
    * Install the various python versions listed in tox.ini.
        * *UNIX* Pyenv is probaly the easiest way to do this, set all of the tox tested versions
        as global so tox can find them, e.g.:
        ```sh
        pyenv global 3.9.0 3.8.0 3.7.0 3.6.1
        ```
        * *Windows* You could try pyenv-win but I don't know how well it plays with tox. Easier is
        to install all the python versions via .exe installers from https://www.python.org/downloads/windows/
        Make sure they are all added to PATH.
    * Git clone the repo into a new directory/folder
    * Use pip to install tox and run it from the cloned repo

## Latex Documentation with Sphinx and Overleaf
NLoed uses [Sphinx](https://www.sphinx-doc.org/en/master/) to automatically generate its function documentation that describes the package
contents and call structures in detail. Sphinx's output is a latex project which needs to be compiled
in order to produce a PDF. A number of other latex projects are also included in NLoed's docs folder.
NLoed's Github repo can be automatically linked to an Overleaf account to compile these latex projects.
Sphinx will already be installed along with the other development dependencies from requirments.txt.

1. **Using Sphinx**
    Sphinx is used to extract docstrings automatically from the NLoed source code. Sphinx can then generate
    a latex project that can be compiled into a pdf listing all of the class function and call structures
    documented in the NLoed source code. The exact content of the resulting document is controlled by
    the .rst files in docs/latex_sphinx_manual/source. The docs folder also contains other latex projects for
    NLoed's other documentation. 
    * The cloned NLoed repo should be able to run Sphinx by default. To (re)make the Sphinx latex project after any updates you can run in the latex_sphinx_manual directory:
        ```sh
        make latex
        ```
        You need to run this after updating any docstrings or making changes to the .rst files.
    * (Optional) If you've already made the latex project once (or you've cloned the repo and are
        unsure of the Sphinx latex project status, you can run the following command to clean up the latex
        buid (it must be run in the latex_sphinx_manual directory):
        ```sh
        make latex
        ```
    * (Optional) If you add new modules to the NLoed project you will need to re-run the autodoc extension of Sphinx so that it can find the new code and docstrings. This will modify both nloed.rst and modules.rest (although the later isn't used by default). To do this run in the repo's root directory:
        ```sh
        $ sphinx-apidoc -o docs/latex_sphinx_manual/source nloed
        ```

2. **Linking and Compiling in Overleaf**
    The latex projects in NLoed's docs folder need to be compiled in order to generate readable PDFs
    that can (and should) be updated and included in the Github repository. Compiling these documents on
    your machine can be a pain, unless you've settup an easy way to install all of the required latex
    packages. To make things easier you can link NLoed's Github repo to an Overleaf account and overleaf

    * You can link Github and Overleaf by following these guides:
        * https://www.overleaf.com/learn/how-to/Using_Git_and_GitHub
        * https://www.overleaf.com/learn/how-to/How_do_I_connect_an_Overleaf_project_with_a_repo_on_GitHub,_GitLab_or_BitBucket%3F
    * Depending how you've set up the Github-Overleaf link, you may have the multiple latex projcts imported from Github into the same Overleaf project. If is this case you can manually set the main tex file of each project to be the 'Main document' in Overleaf using the 'Menu' dropdown in the top right. Select each tex file individually and hit 'Recompile' and then download the resulting PDF. Repeat this for each project to render all of the PDFS.
s

## Notebook Examples and the Example Pack

The examples folder contains a collection of projects demonstrating some of the basic uses of NLoed.
Each project has its own folder within the examples folder; projects currently include demonstrations
on a hill function model, multiple linear regression, a multi-input/output non-normal model (similar
to a generalized linear model), and a nonlinear ODE. 

The example projects consists of a set of Jupyter notebook (.pynb) files, each devoted to a specific
task within the project. For example there are seperate notebooks for creating a model, designing 
experiments and fitting a model. This is intended to provide new users with a modular and incremental
introduction to using NLoed. 

At the time of writing, Github will render these examples as notebooks in the online repository.
This makes it useful for those browsing the repository to get a sense for how the package is used. 
In order to run the notebooks on your local machine you will first need to clone the repository
or download the examples folder and ensure that NLoed is installed. Installing Jupyter notebook can 
be done using pip via the command:
```sh
pip install notebook
```
Then, from the example folder or a given project folder, start a Jupyter notebook instance using:
```sh
jupyter notebook
```
This will open a web browser tab which will allow you to navigate the project folders and open the desired
notebook file. The notebook code can be run using the standard Jupyter notebook commands in the browser.

Regular python scripts (.py) are also provided for each notebook in a project. 
Note, that the later notebooks in a project depend on the objects generated in earlier notebooks.
(The notebooks are named starting with 'N#_' where the number '#' indicates the order they would 
normally be run in.) In each notebook after the first, the notebook starts with a call to the regular
python script version of the previous notebook in order to load the previously generated objects 
into the notebook session of the current notebook. This is done automatically and so, for existing
notebooks, user do not need to worry about the dependence between notebooks. However, when writing
new examples developers should follow the established naming convention (using the 'N#_' prefix), and
generate python scripts for each notebook in a project by running the following command in the project
directory after they have created all of their project notebooks:
```sh
jupyter nbconvert --to script *.ipynb
```
This command will create a regular python script (.py) from all the notebooks (.pynb)
Previous notebooks can then be linked to a current notebook by adding the following command to the top
 of the current notebook:
 ```sh
 from N#_previous_notebook_name import *
 ```

## Packaging and Release
In order to make NLoed available using pip, it needs to be packaged properly and uploaded to the
PyPI repository. This involves creating a source and built distribution, followed uploading the resulting files.
It is also a good idea to tag all releases to PyPI on the Github repo.

1. **Creating a Source Distribution using sdist**
    Source distributions bundles the source code into a .tar.gz, basic but not fast and may contain
    more than is needed (this is controlled by the MANIFEST.in file)
    * Create a Source Distribution:
        ```sh
        python setup.py sdist
        ```

2. **Creating a Built Distribution using Wheels**
    Wheels distrobutions are built and ready to install, pip uses these by default. 
    * Create a Wheels Built Distribution:
        ```sh
        python setup.py bdist_wheel
        ```

3. **Uploading to PyPI using Twine**
    PyPI is the repository where pip installs packages from. [Twine](https://twine.readthedocs.io/en/latest/) is the tool used to upload python
    packages to the repository. In order to update the NLoed package on PyPI the user must have the 
    appropriate credentials (i.e. password or key). PyPI also maintains a test repository for experimenting
    with the packaging process. The standard and test repo are completely indpendent and may not
    have the same packages or version and require seperate accounts.

    * Upload NLoed to the PyPI Repository:
        ```sh
        twine upload dist/*
        ```
    * (Optional) Upload NLoed to the PyPI Test Repository:
        ```sh
        twine upload --repository testpypi dist/*
        ```
    * (Optional) Installing from the PyPI Test Repository:
        ```sh
        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nloed
        ```
        (dependencies are install from the regular pip repo)

    4. **Tagging a Release on Github**
        Marking the version of the code on Github that is uploaded on PyPI is a good idea for tracking which
        releases map to which versions in the git history. See Github docs for details.
