#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python wrapper for SIR
"""
from distutils.ccompiler import CCompiler
from distutils.errors import DistutilsExecError, CompileError
from distutils.unixccompiler import UnixCCompiler
from setuptools import find_packages, setup
from setuptools.extension import Extension

import os
import platform
from subprocess import Popen, PIPE
import sys
import numpy


def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    compiler_so = self.compiler_so
    
    # Turn off machine-specific optimizations
    compiler_so = [x for x in compiler_so if "marc" not in x and "mtune" not in x]
        
    arch = platform.architecture()[0].lower()
    if (ext == ".f" or ext == ".f90"):
        if sys.platform == 'darwin' or sys.platform.startswith('linux'):
            compiler_so = [os.environ['FC']]
            if (ext == ".f90"):
                cc_args = ["-O3", "-fPIC", "-c", "-ffree-form", "-ffree-line-length-none"]
            if (ext == ".f"):
                cc_args = ["-O3", "-fPIC", "-c", "-fno-automatic", "-ffixed-line-length-none"]            
            # Force architecture of shared library.
            if arch == "32bit":
                cc_args.append("-m32")
            elif arch == "64bit":
                cc_args.append("-m64")
            else:
                print("\nPlatform has architecture '%s' which is unknown to "
                      "the setup script. Proceed with caution\n" % arch)
                    
    try:
        self.spawn(compiler_so + cc_args + [src, '-o', obj] +
                   extra_postargs)
    except DistutilsExecError as msg:
        raise CompileError(msg)
UnixCCompiler._compile = _compile


# Hack to prevent build_ext from trying to append "init" to the export symbols.
class finallist(list):
    def append(self, object):
        return


class MyExtension(Extension):
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self.export_symbols = finallist(self.export_symbols)


def get_libgfortran_dir():
    """
    Helper function returning the library directory of libgfortran. Useful
    on OSX where the C compiler oftentimes has no knowledge of the library
    directories of the Fortran compiler. I don't think it can do any harm on
    Linux.
    """
    for ending in [".3.dylib", ".dylib", ".3.so", ".so"]:
        try:
            p = Popen(['gfortran', "-print-file-name=libgfortran" + ending],
                      stdout=PIPE, stderr=PIPE)
            p.stderr.close()
            line = p.stdout.readline().decode().strip()
            p.stdout.close()
            if os.path.exists(line):
                return [os.path.dirname(line)]
        except:
            continue
        return []

# Monkey patch the compilers to treat Fortran files like C files.
CCompiler.language_map['.f90'] = "c"
UnixCCompiler.src_extensions.append(".f90")
CCompiler.language_map['.f'] = "c"
UnixCCompiler.src_extensions.append(".f")

# with open("VERSION.txt", "rt") as fh:
  # VERSION = fh.read().strip()
DOCSTRING = __doc__.strip().split("\n")

pathGlobal = "bayesclumpy/src/"

listFiles = [pathGlobal+'interface.f90', pathGlobal+'nenkova.pyx']

libBC = MyExtension('bayesclumpy.codes.nenkova',
                  libraries=["gfortran"],
                  library_dirs=get_libgfortran_dir(),
                  sources=listFiles,
                  include_dirs=[numpy.get_include()])


setup_config = dict(
    name='bayesclumpy',
    version='2.0',
    description=DOCSTRING[0],
    long_description="\n".join(DOCSTRING[2:]),
    author=' A. Asensio Ramos',
    author_email='aasensio@iac.es',
    url='https://github.com/aasensio/pyiacsun',
    license='GNU General Public License, version 3 (GPLv3)',
    platforms='OS Independent',
    install_requires=[
        'numpy',
    ],
    ext_modules=[libBC],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: Implementation :: CPython",
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords=['Bayesclumpy'],
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    package_data={'bayesclumpy': [os.path.join('nenkova', '*')]}
)

if __name__ == "__main__":
    setup(**setup_config)
    try:
        os.remove("bc.mod")
    except:
        pass
