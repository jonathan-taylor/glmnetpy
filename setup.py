from setuptools import setup, Extension
import pybind11
import os


# find eigen source directory of submodule

dirname = os.path.abspath(os.path.dirname(__file__))
eigendir = os.path.abspath(os.path.join(dirname, 'eigen'))

module = Extension(
    'glmnetpp',
    sources=['src/wls_exp.cpp'],
    include_dirs=[pybind11.get_include(),
                  eigendir,
                  'src/glmnetpp/include',
                  'src/glmnetpp/src',
                  'src/glmnetpp/test'],
    language='c++',
    extra_compile_args=['-std=c++17']
)

setup(
    name='glmnetpp',
    version='0.1',
    ext_modules=[module]
)

