# pyglmnet
Python bindings to glmnet base source


## Organization

- `src` contains `glmnet` C++ code for the two main routines
  `spwls_exp` and `wls_exp`.
- `tests` contains test data from R so that we can verify we get the
  same results in python. Also contains some code for regenerating
  data using an instrumented version of glmnet.
  

## Steps to build

- Assuming Xcode with command line tools is installed already.

- Also probably a good idee to test in a new virtual env

### Create a new virtual env, say `pyg`.

```
conda create -n pyg python=3.10 -y
conda activate pyg
```

### Install requirements

```
pip install -r requirements.txt
```

#### Specifying `Eigen`

By default, build assumes the top level directory contains a directory `eigen` with the `Eigen` headers. This
can be achieved with

```
git clone https://github.com/libigl/eigen.git --depth 5
```

Alternatively, if you have `Eigen` installed elsewhere, set the environment variable `EIGEN_LIBRARY_PATH` to
the appropriate path.

### Build the package

```
pip install .
```

Load the module:

```
import glmnet.glmnetpp as gpp
```

This provides two functions:

- `gpp.wls_exp` for dense $x$
- `gpp.spwls_exp` for sparse $x$

### Testing

Example invocations can be seen in `tests/test_pyglmnet.py`. So one can run all the tests via:
   
```
pytest tests
```

No error messages mean all is well.







