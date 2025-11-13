import numpy as np
from astropy.io import fits
from scipy.integrate import quad

f = fits.open('circinus_bc.fits')

sigma = f[0].data[:, 1]
N0 = f[0].data[:, 2]
out = np.zeros(len(sigma))
err = np.zeros(len(sigma))

for i in range(len(sigma)):    
    integ = lambda x: np.cos(np.pi / 180.0 * x) * np.exp(-N0[i] * np.exp(-x**2 / sigma[i]**2))
    out[i], err[i] = quad(integ, 0.0, 90.0)

out = 1.0 - out