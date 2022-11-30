import bayesclumpy
import numpy as np
import matplotlib.pyplot as pl

if (__name__ == '__main__'):

    # Sampling
    bc = bayesclumpy.Bayesclumpy('conf_hoenig.ini', verbose=1)
    bc.sample()
    
    bc.corner('circinus_bc_hoenig.fits')
    bc.posterior_check('circinus_bc_hoenig.fits')
