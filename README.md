# glmnet


## Specifying `Eigen`

By default, build assumes the top level directory contains a directory `eigen` with the `Eigen` headers. The current stable
version is 3.4.0, with which the library is build:
can be achieved with

```
git clone --branch 3.4.0  https://gitlab.com/libeigen/eigen.git --depth 5
```

Alternatively, if you have `Eigen` installed elsewhere, set the environment variable `EIGEN_LIBRARY_PATH` to
the appropriate path.

### Build the package

```
pip install .
```








