project(
  'bayesclumpy2',
  'c', 'fortran',
  version: '2.0',
  meson_version: '>=1.1.0',
)

py = import('python').find_installation(pure: false)
fc = meson.get_compiler('fortran')

# The Fortran sources have lines longer than the standard free-form limit.
fortran_args = fc.get_supported_arguments('-ffree-line-length-none')

incdir_numpy = run_command(
  py, '-c', 'import numpy; print(numpy.get_include())',
  check: true,
).stdout().strip()
numpy_dep = declare_dependency(compile_args: ['-I' + incdir_numpy])

# Cython is run explicitly rather than declared as a project language: Meson's
# Cython sanity check links against whichever python3 pkg-config reports, which
# can be a system Python unrelated to the one being built for.
nenkova_c = custom_target(
  'nenkova_c',
  input: 'bayesclumpy/src/nenkova.pyx',
  output: 'nenkova.c',
  command: [find_program('cython'), '-3', '@INPUT@', '-o', '@OUTPUT@'],
)

py.extension_module(
  'nenkova',
  nenkova_c,
  'bayesclumpy/src/interface.f90',
  fortran_args: fortran_args,
  dependencies: numpy_dep,
  link_language: 'fortran',
  subdir: 'bayesclumpy/codes',
  install: true,
)

py.install_sources(
  'bayesclumpy/__init__.py',
  'bayesclumpy/bayesclumpy.py',
  'bayesclumpy/censoring.py',
  'bayesclumpy/model_encoder.py',
  'bayesclumpy/model_nn.py',
  'bayesclumpy/models.py',
  'bayesclumpy/observations.py',
  'bayesclumpy/tools.py',
  subdir: 'bayesclumpy',
)

# Data files read at runtime, relative to the installed package.
foreach data_dir : ['filters', 'nenkova', 'neural']
  install_subdir(
    'bayesclumpy' / data_dir,
    install_dir: py.get_install_dir() / 'bayesclumpy',
  )
endforeach
