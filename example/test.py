import bayesclumpy
import numpy as np


if (__name__ == '__main__'):

    # Sampling
    bc = bayesclumpy.Bayesclumpy('conf.ini')
    # bc.sample()
    
    bc.corner('circinus_bc.fits')
    bc.posterior_check('circinus_bc.fits')