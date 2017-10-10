# Attainable Region (AR) Tools

This is the GitHub repository of Attainable Region (AR) theory. The main homepage of AR theory can be found at [attainableregions.com](http://attainableregions.com "AR theory homepage").

This repository contains:
1. A collection of Python functions for carrying out AR computations, `artools.py`
2. Interactive Jupyter notebooks that describe the basics of AR theory
3. Source code for the examples found on attainableregions.com

## 1. AR tools
`artools.py` is a collection of Python functions for carrying out AR calculations as well as performing simple 2-D and 3-D visualisations of convex regions. A Jupyter notebook demonstrating its usage can be found in the [notebooks folder](./AR-book/AR-book-notebooks/Ch%208/artools%20demos.ipynb).

### 1.1 Usage
Copy `artools.py` to your local directory and import.
```python
import artools
```

Alternatively, add the directory containing `artools.py` to your system path, and then import.
```python
import sys
sys.path.append('PATH/TO/ARTOOLS/DIRECTORY')

import artools
```

### 1.2 Dependencies
Currently, `artools.py` is written in **Python 3** and depends on [SciPy](https://www.scipy.org/) and [Matplotlib](https://matplotlib.org/). We recommend using [Anaconda](https://www.continuum.io/downloads) to manage the installation of these packages.

The code currently supports Python versions 3.4 - 3.6.

## 2. Jupyter notebooks
<img src="./AR-book/notebook_demo.gif" width="480">

Jupyter notebooks, which demonstrate common AR theory concepts, can be found [here](./AR-book/AR-book-notebooks). These notebooks are written in conjunction with the [Attainable Region theory textbook](http://eu.wiley.com/WileyCDA/WileyTitle/productCd-1119157889.html).

### 2.1 If you are using virtual environments
Notebook widgets might not display if Jupyter is run in a virtual environment. If this is the case, issue the following console command:
```bash
jupyter nbextension enable --py widgetsnbextension
```

See this [Stack Overflow post](https://stackoverflow.com/questions/36351109/ipython-notebook-ipywidgets-does-not-show) and [ipywidgets Github installation instructions](https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/user_install.md) for more details.

## 3. AR website examples
The source code for the examples described on attainableregions.com are stored [here](./AR-book/website-examples).

## NOTES
- As of 10 Oct 2017, Python 2 will no longer be supported. All code from now on will support Python 3.4 - 3.6 and may not work with any other versions of Python.
- Please use [2to3.py](https://docs.python.org/3.0/library/2to3.html) to convert Python 2 `.py` files to Python 3, and [2to3_nb.py](https://github.com/pycam/python-intro/blob/master/install/2to3_nb.py) to convert Jupyter `.ipynb` notebooks.
