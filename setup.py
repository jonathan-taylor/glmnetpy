from setuptools import setup, Extension
import pybind11
import os


eigendir = os.path.abspath(os.path.join('.', 'eigen'))

print(eigendir, 'eigen dir')
print(os.listdir(eigendir))
print(__file__)
print(os.listdir('.'))
print(os.listdir(os.path.join(eigendir, '..')))
print(os.listdir(os.path.join(eigendir, '..', 'eigen')))

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

