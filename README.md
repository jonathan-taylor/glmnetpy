# pyglmnet
Python bindings to glmnet base source


## Organization

- `src` contains `glmnet` C++ code for the two main routines
  `spwls_exp` and `wls_exp`.
- `tests` contains test data from R so that we can verify we get the
  same results in python. Also contains some code for regenerating
  data using an instrumented version of glmnet.
  

## Steps to build

Assuming Xcode with command line tools is installed already.

### Create a new virtual env, say `pyg`.

```
conda create -n pyg python=3.10 -y
```

### Install requirements

```
pip install -r requirements.txt
git submodule update --init # fetches Eigen as a git submodule
```

### Build the package

```
python setup.py build_ext --inplace
```

This will produce a shared library in the current directory, after which one should be able to load the module:

```
import glmnetpp as gpp
```

This provides two functions:

- `gp.wls_exp` for dense $x$
- `gp.spwls_exp` for sparse $x$

4. Example invocations can be seen in `test_pyglmnet.py`. So one can
   run all the tests via:
   
```
python test_pyglmnet.py
```

No error messages mean all is well.







