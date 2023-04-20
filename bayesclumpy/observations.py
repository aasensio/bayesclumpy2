import numpy as np
from scipy.stats import norm
from scipy.optimize import root
from scipy.special import erf

def fun(s, l, confidence):
    """
    This function is optimized to compute the optimal standard deviation in the case
    of upper limits. Upper limits are given as "the flux is below l with confidence (%)".
    In the case of censored data, the likelihood of a lower censored data point is
    p(f<l) = Phi((l-f) / sigma) when sigma is known. Since we don't know sigma but
    we give a confidence interval, we integrate the likelihood from 0 to l and compute
    sigma forcing the integral to be equal to the confidence.
    The integrals int(p(f<l) df=0...l) and int(p(f<l) df=0...inf) can be computed
    analytically in Mathematica.

    Parameters
    ----------
    s : float
        Standard deviation of the noise
    l : float
        Lower limit
    confidence : float
        [description]

    Returns
    -------
    [type]
        [description]
    """
    sqrt2 = np.sqrt(2.0)
    sqrtpi = np.sqrt(np.pi)
    cdf_integral_all = 0.5 * erf(1.0/(sqrt2 * s)) + 0.5 + s / (sqrt2 * sqrtpi) * np.exp(-1.0/(2*s**2))
    cdf_integral = l * erf(l/(sqrt2 * s)) + s / (sqrt2 * sqrtpi) * (np.exp(-l**2/(2*s**2)) - 1.0)

    return cdf_integral / cdf_integral_all - confidence
    
class Observation(object):

    def __init__(self, filename=None, verbose=False):
        
        self.verbose = verbose

        if filename is not None:
            self.filename = filename
            self.read_file(self.filename)

    def read_file(self, filename):
                
        with open(filename, 'r') as f:
            self.name = f.readline().replace("'", "").rstrip()
            tmp = f.readline().split()
            self.n_filters, self.n_spec = int(tmp[0]), int(tmp[1])
            
            if (self.n_spec == 0):
                self.transmission = [None] * self.n_filters
                self.filter = [None] * self.n_filters
                self.obs_flux = [None] * self.n_filters
                self.obs_sigma = [None] * self.n_filters
                self.type_uncertainty = [None] * self.n_filters
                self.type = [None] * self.n_filters
                self.obs_wave = [None] * self.n_filters
            else:
                self.transmission = [None] * (self.n_filters + 1)
                self.filter = [None] * (self.n_filters + 1)
                self.obs_flux = [None] * (self.n_filters + 1)
                self.obs_sigma = [None] * (self.n_filters + 1)
                self.type_uncertainty = [None] * (self.n_filters + 1)
                self.type = [None] * (self.n_filters + 1)
                self.obs_wave = [None] * (self.n_filters + 1)

            loop = 0
            # Read photometry
            for i in range(self.n_filters):
                line = f.readline()
                tmp = line.split()
                filter = tmp[0].replace("'", "")
                self.filter[loop] = filter

                flux = tmp[1]
                sigma = tmp[2]

                self.type[loop] = 'phot'

                # If the measurement is an upper limit, given a confidence level (0.68, 0.95, etc.)
                # the integral between 0 and the tabulated value is 0.68, 0.95, etc. times
                if ('<' in flux):
                    self.type_uncertainty.append('upper')
                    flux = float(flux.replace('<', ''))
                    sigma = float(sigma)
                    
                    s0 = np.sqrt(flux)
                    tmp = root(fun, s0, args=(flux, sigma))
                    
                    self.obs_flux[loop] = flux
                    self.obs_sigma[loop] = tmp.x[0]
                else:
                    self.type_uncertainty[loop] = 'normal'
                    self.obs_flux[loop] = float(flux)
                    self.obs_sigma[loop] = float(sigma)

                loop += 1

            # Read spectroscopy if present
            if (self.n_spec != 0):
                line = f.readline()
                tmp = line.split()
                filter = tmp[0].replace("'", "")
                self.filter[loop] = filter
                self.type[loop] = 'spec'
                self.type_uncertainty[loop] = 'normal'

                wave = np.zeros(self.n_spec)
                flux = np.zeros(self.n_spec)
                sigma = np.zeros(self.n_spec)

                for i in range(self.n_spec):
                    line = f.readline()
                    tmp = line.split()

                    wave[i] = float(tmp[0])
                    flux[i] = float(tmp[1])
                    sigma[i] = float(tmp[2])
                
                self.obs_flux[loop] = flux
                self.obs_sigma[loop] = sigma
                self.obs_wave[loop] = wave       

        if (self.verbose):
            print(f'OBSERVATIONS')
            print(f'Name : {self.name}')
            if (self.n_filters > 0):
                print(f'Photometry : ')
        
        for ifilter in range(self.n_filters):
            filter = self.filter[ifilter]
            
            path = str(__file__).split('/')
            root = '/'.join(path[0:-1])
            f = open(f'{root}/filters/{filter}.res', 'r')
            tmp = f.readline().split()
            n_lambda = int(tmp[0])
            normalization = float(tmp[1])
            
            T = np.zeros((2, n_lambda))
            for i in range(n_lambda):
                tmp = f.readline().split()
                T[0, i] = float(tmp[0])
                T[1, i] = float(tmp[1])

            # Set to zero very small or negative values of the transmission
            T[1, T[1, :] < 1e-5] = 0.0

            # Reorder arrays to force ascending wavelength
            ind = np.argsort(T[0, :])
            T = T[:, ind]

            # Area normalization
            area = np.trapz(T[1, :], x=T[0, :])
            T[1, :] /= area

            self.transmission[ifilter] = T

            # Compute central wavelength
            self.obs_wave[ifilter] = np.trapz(T[0, :] * T[1, :], x=T[0, :])

            if (self.verbose):
                if (self.type[ifilter] == 'phot'):
                    if (self.type_uncertainty[ifilter] == 'normal'):
                        print(f"  * {filter:10s} -> flux       ={self.obs_flux[ifilter]:8.2f} - sigma   ={self.obs_sigma[ifilter]:8.2f} - lambda0={self.obs_wave[ifilter]:6.2f} \u03BCm - {self.type_uncertainty[ifilter]}")
                    if (self.type_uncertainty[ifilter] == 'upper'):
                        print(f"  * {filter:10s} -> upper limit={self.obs_flux[ifilter]:8.2f} - eq.sigma={self.obs_sigma[ifilter]:8.2f} - lambda0={self.obs_wave[ifilter]:6.2f} \u03BCm - {self.type_uncertainty[ifilter]}")

        if (self.verbose):
            if (self.n_spec != 0):
                print(f'Spectroscopy : ')
                print(f"  * {self.filter[0]:10s} -> n = {self.n_spec}")

if __name__ == '__main__':

    tmp = Observation(filename='observations/circinus_upper.dat', verbose=True)

