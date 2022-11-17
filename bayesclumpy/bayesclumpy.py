import numpy as np
from configobj import ConfigObj
from .observations import Observation
from .models import Models
from .tools import _lower_to_sep
from .gui import inspect
from tqdm import tqdm
from astropy.io import fits
import corner
import matplotlib.pyplot as pl

try:
    from dynesty import NestedSampler
    DYNESTY_PRESENT = True
except:
    DYNESTY_PRESENT = False
try:
    import emcee
    EMCEE_PRESENT = True
except:
    EMCEE_PRESENT = False

try:
    import ultranest
    ULTRANEST_PRESENT = True
except:
    ULTRANEST_PRESENT = False

try:
    import nestle
    NESTLE_PRESENT = True
except:
    NESTLE_PRESENT = False


class Bayesclumpy(object):

    def __init__(self, conf_file, verbose=True):
        self.verbose = verbose
        self.conf_file = conf_file

        # Check available algorithms
        self.algorithms = []
        print("SAMPLERS")
        if (DYNESTY_PRESENT):
            self.algorithms.append('dynesty')
            if (verbose):
                print(" - Nested sampling (found)")
        if (EMCEE_PRESENT):
            self.algorithms.append('emcee')
            if (verbose):
                print(" - emcee (found)")
        if (ULTRANEST_PRESENT):
            self.algorithms.append('ultranest')
            if (verbose):
                print(" - ultranest (found)")
        if (NESTLE_PRESENT):
            self.algorithms.append('nestle')
            if (verbose):
                print(" - nestle (found)")

        if (len(self.algorithms) == 0):
            raise Exception("No sampler available.")

        self.config = self.read_configuration(conf_file)
        
        self.obs = Observation(filename=self.config['observations']['file'], verbose=verbose)
        self.models = Models(self.config, self.obs, verbose=verbose)

    def read_configuration(self, conf_file):
        f = open(conf_file, 'r')
        tmp = f.readlines()
        f.close()

        self.configuration_txt = tmp

        input_lower = ['']

        for l in tmp:
            input_lower.append(_lower_to_sep(l)) # Convert keys to lowercase

        # Parse configuration file        
        return ConfigObj(input_lower, list_values=False)
    
    def sample(self):        
        sampler_name = self.config['general']['sampler']
        
        if (self.verbose):
            print(f"SAMPLING with {sampler_name}")

        if sampler_name not in self.algorithms:
            raise Exception(f'Desired sampler ({sampler_name}) not implemented or incorrect.')

        if (sampler_name == 'dynesty'):
            self.sampler = NestedSampler(self.models.loglike, self.models.prior_transform_nested, self.models.n_parameters, bound='multi')
            self.sampler.run_nested()
            self.samples = self.sampler.results.samples

        if (sampler_name == 'ultranest'):
            self.sampler = ultranest.ReactiveNestedSampler(self.models.parameter_names, self.models.loglike, self.models.prior_transform_nested)
            self.samples = self.sampler.run(region_class=ultranest.mlfriends.RobustEllipsoidRegion)

        if (sampler_name == 'nestle'):
            self.sampler = nestle.sample(self.models.loglike, self.models.prior_transform_nested, self.models.n_parameters, method='multi', callback=nestle.print_progress)
            self.samples = nestle.resample_equal(self.sampler.samples, self.sampler.weights)
            
        if (sampler_name == 'emcee'):
            
            # Initial solution (center of the interval)
            center = np.array([self.models.central[key] for key in self.models.parameter_names])
            if (self.verbose):
                print(f"Initial solution : {center}")
            self.nwalkers = 32            
            pos = center + 1e-4 * np.abs(np.random.randn(self.nwalkers, self.models.n_parameters))
            
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.models.n_parameters, self.models.loglike_emcee)
            self.sampler.run_mcmc(pos, 500, progress=True)

            self.samples = self.sampler.get_chain(discard=50, thin=5, flat=True)

            for i, (k, v) in enumerate(self.models.typical.items()):
                self.samples[:, i] *= self.models.typical[k]

        self.get_samples_sed()

    def get_samples_sed(self):
        n_samples = self.samples.shape[0]
        seds = np.zeros((n_samples, len(self.models.base_wavelength)))
        for i in tqdm(range(n_samples)):            
            seds[i, :] = self.models.loglike(self.samples[i, :], only_synth=True)

        header = fits.Header()
        for k, v in self.config.items():
            for k2, v2 in self.config[k].items():
                label = k2.replace(' ','')
                header[label[0:8]] = f'{v2}'

        hdu1 = fits.PrimaryHDU(self.samples, header=header)
        hdu2 = fits.ImageHDU(self.models.base_wavelength)
        hdu3 = fits.ImageHDU(seds)        
        hdul = fits.HDUList([hdu1, hdu2, hdu3])
        hdul.writeto(self.config['general']['output file'], overwrite=True)

        if (self.verbose):
            print(f"\nSAVING results on {self.config['general']['output file']}")

    def inspect(self):
        tmp = np.loadtxt('sample.sample')
        inspect(tmp)

    def corner(self, bc_file):

        self.f = fits.open(bc_file)

        samples = self.f[0].data

        corner.corner(samples, 
            smooth=1, 
            smooth1d=1, 
            labels = self.models.parameter_names, 
            quantiles=(0.16, 0.84), 
            levels=(0.68,)
            )

    def posterior_check(self, bc_file):

        self.f = fits.open(bc_file)

        wave = self.f[1].data
        seds = self.f[2].data

        fig, ax = pl.subplots()
        ax.plot(wave, seds.T, alpha=0.02, color='C1')
        ax.errorbar(self.obs.obs_wave[0:self.obs.n_filters], self.obs.obs_flux[0:self.obs.n_filters], fmt='o', capsize=3, elinewidth=2, yerr=self.obs.obs_sigma[0:self.obs.n_filters], color='C0')
        if (self.obs.n_spec != 0):
            ax.errorbar(self.obs.obs_wave[-1], self.obs.obs_flux[-1], fmt='o', capsize=3, elinewidth=2, yerr=self.obs.obs_sigma[-1], color='C2')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(u'Wavelength [\u03bcm]')
        ax.set_ylabel(u'Flux [Jy]')


        