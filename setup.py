import os

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')
import setuptools

from setuptools import setup, Extension
import pybind11
import versioneer
from cythexts import cyproc_exts, get_pyx_sdist

# Get various parameters for this version, stored in ISLP/info.py

class Bunch(object):
    def __init__(self, vars):
        for key, name in vars.items():
            if key.startswith('__'):
                continue
            self.__dict__[key] = name

def read_vars_from(ver_file):
    """ Read variables from Python text file

    Parameters
    ----------
    ver_file : str
        Filename of file to read

    Returns
    -------
    info_vars : Bunch instance
        Bunch object where variables read from `ver_file` appear as
        attributes
    """
    # Use exec for compabibility with Python 3
    ns = {}
    with open(ver_file, 'rt') as fobj:
        exec(fobj.read(), ns)
    return Bunch(ns)

info = read_vars_from(os.path.join('glmnet', 'info.py'))

# find eigen source directory of submodule

dirname = os.path.abspath(os.path.dirname(__file__))
eigendir = os.path.abspath(os.path.join(dirname, 'eigen'))

if 'EIGEN_LIBRARY_PATH' in os.environ:
    eigendir = os.path.abspath(os.environ['EIGEN_LIBRARY_PATH'])

# get long_description

long_description = open('README.md', 'rt', encoding='utf-8').read()
long_description_content_type = 'text/markdown'

EXTS = [Extension(
    f'glmnet.{mod}',
    sources=[f'src/{mod}_exp.cpp',
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
                                                   'multigaussnet'][:1]]

# Cox extension

EXTS.append(Extension('glmnet.cox_utils',
                      ['src/cox/cox_utils.pyx', 'src/cox/cox_fns.c'],
                      include_dirs=['src/cox']))


# Add numpy includes when building extension.
build_ext, need_cython = cyproc_exts(EXTS,
                                     '3.0',
                                     'pyx-stamps')
def make_np_ext_builder(build_ext_class):
    """ Override input `build_ext_class` to add numpy includes to extension

    This is useful to delay call of ``np.get_include`` until the extension is
    being built.

    Parameters
    ----------
    build_ext_class : class
        Class implementing ``distutils.command.build_ext.build_ext`` interface,
        with a ``build_extensions`` method.

    Returns
    -------
    np_build_ext_class : class
        A class with similar interface to
        ``distutils.command.build_ext.build_ext``, that adds libraries in
        ``np.get_include()`` to include directories of extension.
    """
    class NpExtBuilder(build_ext_class):

        def build_extensions(self):
            """ Hook into extension building to add np include dirs
            """
            # Delay numpy import until last moment
            import numpy as np
            for ext in self.extensions:
                ext.include_dirs.append(np.get_include())
            build_ext_class.build_extensions(self)

    return NpExtBuilder
build_ext = make_np_ext_builder(build_ext)

cmdclass = versioneer.get_cmdclass()
cmdclass=versioneer.get_cmdclass()
cmdclass.update(dict(
    build_ext=build_ext,
    sdist=get_pyx_sdist()))

def main(**extra_args):
    setup(name=info.NAME,
          maintainer=info.MAINTAINER,
          maintainer_email=info.MAINTAINER_EMAIL,
          description=info.DESCRIPTION,
          url=info.URL,
          download_url=info.DOWNLOAD_URL,
          license=info.LICENSE,
          classifiers=info.CLASSIFIERS,
          author=info.AUTHOR,
          author_email=info.AUTHOR_EMAIL,
          platforms=info.PLATFORMS,
          version=versioneer.get_version(),
          packages = ['glmnet'],
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


