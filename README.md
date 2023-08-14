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

0. Create a new virtual env, say `pyg`.

1. Install [`pybind11`](https://pybind11.readthedocs.io).

2. Install the [Eigen
   library](https://eigen.tuxfamily.org/index.php). Easily done with
   homebrew.
   ```
   brew install eigen
   ```
3. If all pre-reqs are met,  the following should build the module.

```
./build.sh
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







