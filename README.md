# glmnet


## Specifying `Eigen`

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








