import os

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')
import setuptools

from setuptools import setup, Extension
import pybind11
import versioneer
#from cythexts import cyproc_exts, get_pyx_sdist

# find eigen source directory of submodule

dirname = os.path.abspath(os.path.dirname(__file__))
eigendir = os.path.abspath(os.path.join(dirname, 'eigen'))

if 'EIGEN_LIBRARY_PATH' in os.environ:
    eigendir = os.path.abspath(os.environ['EIGEN_LIBRARY_PATH'])

cmdclass = versioneer.get_cmdclass()

# get long_description

long_description = open('README.md', 'rt', encoding='utf-8').read()
long_description_content_type = 'text/markdown'

EXTS = [Extension(
    f'glmnet._{mod}',
    sources=[f'src/{mod}.cpp',
             f'src/internal.cpp',
             f'src/update_pb.cpp',
             ],
    include_dirs=[pybind11.get_include(),
                  eigendir,
                  'src/glmnetpp/include',
                  'src/glmnetpp/src',
                  'src/glmnetpp/test'],
    language='c++',
    extra_compile_args=['-std=c++17']) for mod in ['elnet_point',
                                                   'lognet',
                                                   'fishnet',
                                                   'gaussnet',
                                                   'multigaussnet']]

long_description = open('README.md', 'rt', encoding='utf-8').read()

def main(**extra_args):
    setup(name='glmnet',
          version=versioneer.get_version(),
          packages = ['glmnet',
                      'glmnet.paths',
                      'glmnet.test_data'],
          ext_modules = EXTS,
          package_data = {'glmnet':['test_data/*']},
          include_package_data=True,
          data_files=[],
          scripts=[],
          long_description=long_description,
          long_description_content_type=long_description_content_type,
          cmdclass = cmdclass,
          **extra_args
         )

#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main()


