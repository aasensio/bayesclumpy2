# Bayesclumpy v2.0

## Introduction

BayesClumpy is a computer program that can be used for the fast synthesis of spectral 
energy distributions (SED) emerging from clumpy dusty torus models. For the moment,
only the models developed by the Kentucky group are included. 
The fundamental advantage of the code is that these fast synthesis capabilities are used in a 
Bayesian scheme for carrying out inference over the model parameters for observed SED. The code 
is written in standard Fortran 90 and Python.

## Installation

There are different ways to install Bayesclumpy v2.0, but the best is to install it into a 
virtual environment either with pip or conda, which makes everything much more safer, plus 
making sure that all packages are installed for the code. For example, once you have installed 
Miniconda, you can generate a new environment and install the dependencies (you can install 
whatever version of Python 3 you desire). For consistency, it is better to use the gfortran 
compilers from Anaconda. In the following we show how install all packages with Anaconda. 
Anyway, feel free to use the gfortran compiler from your system but you might have some issues.

### For a Linux OS

Within an Anaconda environment, standard packages can be installed with

    conda create -n bayesclumpy python=3.8
    conda activate bayesclumpy
    conda install -c conda-forge cython numpy astropy tqdm scipy gfortran_linux-64 gcc_linux-64 nestle matplotlib configobj pysimplegui corner

### For a Mac OS

Within an Anaconda environment, standard packages can be installed with

    conda create -n bayesclumpy python=3.8
    conda activate bayesclumpy
    conda install -c conda-forge cython numpy astropy tqdm scipy gfortran_osx-64 gcc_osx-64 nestle matplotlib configobj pysimplegui corner

Now install the package into the environment by typing:

    pip install -vv -e .

## Usage

There is an example of usage in the directory `example`. The sampling is done with the following
three lines. First import `bayesclumpy`, instantiate the `Bayesclumpy` class and call the `sample()`
method. This generates a `FITS` file with the MCMC sampling.


    import bayesclumpy
    bc = bayesclumpy.Bayesclumpy('conf.ini')
    bc.sample()

You can then do posterior checks and plot the posterior by using

    bc.corner('output.fits', pdf='corner.pdf')
    bc.posterior_check('output.fits', pdf='posterior_check.pdf')

Everything is controlled from a
human-readable configuration file.

    
    [General]
    Models = Nenkova
    Sampler = nestle 
    Use AGN = False
    Extinction law = chiar06
    Output file = circinus_bc.fits

    [Observations]
    File = observations/circinus.dat

    [Priors]
    Y          = uniform(5,100)
    sigma      = uniform(15,70)
    N          = uniform(1,15)
    q          = uniform(0,3)
    tauv       = uniform(10,150)
    i          = uniform(0,90)
    shift      = uniform(-3,3)
    extinction = range(0,8) uniform(0,5)
    redshift   = uniform(0.0,0.01)

The `General` section defines:

- `Models`: set of models to use during inference. Only `Nenkova` is admitted now but we will include more models in the near future.
- `Sampler`: specific posterior sampling algorithm [`nestle`/`dynesty`/`ultranest`]. At least
one of them should be installed. In our experience, the nested sampler `nestle` works
nicely.
- `Use AGN`: include AGN spectra or not
- `Extiction law`: select one among [`no-extinction`/`allen76`/`seaton79`/`fitzpatric86`/`prevot84`/`calzetti00`/`chiar06`/`chiar_galcen06`]
- `Output file`: FITS file used for output.

The `Observations` section defines:

- `File`: file with the observations. See the examples for guidance.

The `Priors` section defines the prior for each parameter of each specific model. Priors can
be [`uniform`/`normal`/`dirac`]. It can also contain a `range` keyword to define the range of
definition of the variable. This should not be used  with those variables that are input to the
models, which have predefined ranges that are hardwired in Bayesclumpy v2.0.

## Observations

The observations are entered in a file like the following one:

    'Circinus'
    8     0
    'nacoJ'         <1.60   0.95    
    'F160W'         4.77    0.7     
    'nacoK'         19      1.9     
    'naco2p42'      31      3.1     
    'nacoLp'        380     38      
    'nacoMp'        1900    190     
    'trecsSi2'      5939    297     
    'trecsQa'       14078   3520

The first line contains the name of the source. The next line contains the
number of filters and spectropcopic data. For filters, one can provide
measured values or upper limits. If measured values are provided, you must
give the measured value and a estimation of the uncertainty (assumed normal).
If an upper limit is given, you indicate it with a `<` and then provide
the confidence value.

## Dependencies

    cython
    numpy
    astropy
    tqdm
    scipy
    corner
    nestle    
    matplotlib
    configobj
    pysimplegui
    corner

## Optional dependencies

    dynesty
    ultranest
