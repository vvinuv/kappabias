#! /usr/bin/env python
# Authors: Vinu Vikraman <vvinuv@gmail.com>
from distutils.core import setup

setup(
    name="kappabias",
    version="0.1-dev",
    description="Estimating bias from observed shear and galaxy distribution",
    maintainer="Vinu Vikraman",
    maintainer_email="vvinuv@gmail.com",
    license="",
    url='https://github.com/vvinuv/kappabias',
    packages=['bias',],
    classifiers=[
        'Intended Audience :: Dark Energy Survey',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Development Status :: 1 - Alpha',
    ],
)
