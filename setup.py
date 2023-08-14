from setuptools import setup, Extension
import pybind11

module = Extension(
    'glmnetpp',
    sources=['src/wls_exp.cpp'],
    include_dirs=[pybind11.get_include(),
                  '/usr/local/include/eigen3',
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

