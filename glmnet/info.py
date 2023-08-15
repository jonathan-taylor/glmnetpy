""" This file contains defines parameters for glmnet that we use to fill
settings in setup.py, the regreg top-level docstring, and for building the docs.
"""

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description  = 'GlmNet (Python)'

# versions
NUMPY_MIN_VERSION='1.7.1'
SCIPY_MIN_VERSION = '0.9'
MATPLOTLIB_MIN_VERSION = '3.3.3'

NAME                = 'glmnet'
MAINTAINER          = "Naras Balasubrimanian, Trevor Hastie, Jonathan Taylor"
MAINTAINER_EMAIL    = ""
DESCRIPTION         = description
LONG_DESCRIPTION    = description
URL                 = "http://github.org/intro-stat-learning/glmnet"
DOWNLOAD_URL        = ""
LICENSE             = "BSD license"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "Naras Balasubrimanian, Trevor Hastie, Jonathan Taylor"
AUTHOR_EMAIL        = ""
PLATFORMS           = "OS Independent"
STATUS              = 'alpha'
PROVIDES            = []
REQUIRES            = ["numpy (>=%s)" % NUMPY_MIN_VERSION,
                       "scipy (>=%s)" % SCIPY_MIN_VERSION,
                       "matplotlib (>=%s)" % MATPLOTLIB_MIN_VERSION,
                       ]

