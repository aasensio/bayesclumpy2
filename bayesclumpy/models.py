import numpy as np
from scipy.stats import truncnorm, uniform 
import re
from scipy.io import FortranFile
from scipy.interpolate import RegularGridInterpolator
from .tools import tobool
from .codes import nenkova

try:
    import torch
    from . import model_encoder
    from . import model_nn
    TORCH_PRESENT = True
except:
    TORCH_PRESENT = False

class Models(object):
    def __init__(self, config, observations, verbose=False):              

        self.verbose = verbose
        self.obs = observations
        self.model_type = config['general']['models'].lower()
        
        # Manage models
        if (self.model_type == 'nenkova'):
            self.parameter_names = ['y', 'sigma', 'n', 'q', 'tauv', 'i', 'shift', 'extinction', 'redshift']
            self.default_limits = {'y': [5,100], 'sigma': [15,70], 'n': [1,15], 'q': [0,3],
                'tauv':[10, 300], 'i': [0,90], 'shift': [-10,10], 'extinction': [0,5], 'redshift': [0,6]}
            self.typical = {'y': 30.0, 'sigma': 30.0, 'n': 5, 'q': 2.0,
                'tauv': 50.0, 'i': 45.0, 'shift': 1.0, 'extinction': 2.0, 'redshift': 0.0}
            self.n_parameters = len(self.parameter_names)
            self._read_nenkova()
            self.coefs = np.zeros(20)

        if (self.model_type == 'hoenig'):
            if (TORCH_PRESENT):
                self.parameter_names = ['a', 'n0', 'h', 'aw', 'theta_w', 'theta_sig', 'f_wd', 'rout', 'angle', 'shift', 'extinction', 'redshift']
                self.default_limits = {'a': [-3,-0.5], 'n0': [5,10], 'h': [0.1,0.5], 'aw': [-2.5,-0.5],
                    'theta_w':[30.0, 45.0], 'theta_sig': [7, 15], 'f_wd': [0.15, 2.25], 'rout': [450.0, 500.0], 'angle': [0, 90], 'shift': [-10,10], 'extinction': [0,5], 'redshift': [0,6]}
                self.typical = {'a': -1.0, 'n0': 7.0, 'h': 0.3, 'aw': -1.5, 'theta_w': 35.0, 'theta_sig': 10.0, 'f_wd': 1.0, 'rout': 475.0, 'angle': 45.0,'shift': 1.0, 'extinction': 2.0, 'redshift': 0.0}
                self.n_parameters = len(self.parameter_names)
                
                self.device = 'cpu'
                
                self._read_hoenig_neural()
                self.min_n = -8.32
                self.max_n = 5.66                
            else:
                raise Exception('Hoenig models cannot be used without PyTorch installed in your system.')

        self.extiction_law = config['general']['extinction law']
        self.use_agn = tobool(config['general']['use agn'])

        if (self.verbose):
            print(f"Extinction law : {self.extiction_law}")
            print(f"Use AGN : {self.use_agn}")
    
        self.n_parameters = len(self.parameter_names)

        # Parse the priors
        self._parse_prior(config['priors'])

        # Read the extinction curves if needed
        if (self.extiction_law == 'chiar06' or self.extiction_law == 'chiar_galcen06'):
            path = str(__file__).split('/')
            root = '/'.join(path[0:-1])
            self.chiar = np.loadtxt(f'{root}/filters/extinction_chiar_tielens2006.dat', skiprows=14, max_rows=258)

    def _extiction_curve(self, model, wavelength, Av):
        n = len(wavelength)

        found = False

        if (model == 'no-extinction'):
            extinction_curve = np.zeros(n)
            found = True

        if (model == 'allen76'):
            extinction_curve = np.zeros(n)
            found = True

        if (model == 'seaton79'):
            invlambda0 = 4.595
            gamm = 1.051
            C1 = -0.38
            C2 = 0.74
            C3 = 3.96
            C4 = 0.26
            invlambda = 1.0 / wavelength            
            extinction_curve = C1 + C2*invlambda + C3 / ((invlambda - invlambda0**2/invlambda)**2 + gamm**2)
            extinction_curve[invlambda >= 5.9] = C1 + C2*invlambda[invlambda >= 5.9] + C3 / ((invlambda[invlambda >= 5.9] - \
                invlambda0**2/invlambda[invlambda >= 5.9])**2 + gamm**2) + C4*(0.539*(invlambda[invlambda >= 5.9]-5.9)**2 + 0.0564*(invlambda[invlambda >= 5.9]-5.9)**3)

            extinction_curve *= Av
            found = True

        if (model == 'fitzpatric86'):
            invlambda0 = 4.608
            gamm = 0.994
            C1 = -0.69
            C2 = 0.89
            C3 = 2.55
            C4 = 0.50
            invlambda = 1.0 / wavelength            
            extinction_curve = C1 + C2*invlambda + C3 / ((invlambda - invlambda0**2/invlambda)**2 + gamm**2)
            extinction_curve[invlambda >= 5.9] = C1 + C2*invlambda[invlambda >= 5.9] + C3 / ((invlambda[invlambda >= 5.9] - \
                invlambda0**2/invlambda[invlambda >= 5.9])**2 + gamm**2) + C4*(0.539*(invlambda[invlambda >= 5.9]-5.9)**2 + 0.0564*(invlambda[invlambda >= 5.9]-5.9)**3)

            extinction_curve *= Av
            found = True

        if (model == 'prevot84'):
            extinction_curve = np.ones(n)
            found = True

        if (model == 'calzetti00'):
            Rv = 4.05

            extinction_curve = np.zeros(n)

            # From 0.12 micron to 0.63 micron
            ind = np.where((wavelength >= 0.12) & (wavelength < 0.63))[0]
            extinction_curve[ind] = 2.659*(-2.156 + 1.509/wavelength[ind] - 0.198/wavelength[ind]**2 + 0.011/wavelength[ind]**3) + Rv

            # From 0.63 micron to 2.20 micron
            ind = np.where((wavelength >= 0.63) & (wavelength < 2.20))[0]
            extinction_curve[ind] = 2.659*(-1.857 + 1.040/wavelength[ind]) + Rv

            # Below 0.12 micron
            C1 = 2.659*(-2.156 + 1.509/0.12 - 0.198/0.12**2 + 0.011/0.12**3) + Rv
            C2 = 2.659*(-2.156 + 1.509/0.11 - 0.198/0.11**2 + 0.011/0.11**3) + Rv
            slope = (C1-C2) / (0.12-0.11)
            zero = C1 - slope * 0.12
            ind = np.where(wavelength < 0.12)[0]
            extinction_curve[ind] = slope * wavelength[ind] + zero

            # Above 2.20 micron
            C1 = 2.659*(-1.857 + 1.040/2.19) + Rv
            C2 = 2.659*(-1.857 + 1.040/2.20) + Rv
            slope = (C1-C2) / (2.19-2.20)
            zero = C1 - slope * 2.19
            ind = np.where(wavelength > 2.20)[0]
            extinction_curve[ind] = slope * wavelength[ind] + zero

            # Avoid negative values
            extinction_curve[extinction_curve < 0] = 0.0

            extinction_curve = extinction_curve * Av  / Rv
            found = True

        if (model == 'chiar06'):            
            extinction_curve = 0.09 * Av * np.interp(wavelength, self.chiar[:, 0], self.chiar[:, 2])            
            found = True

        if (model == 'chiar_galcen06'):            
            extinction_curve = 0.09 * Av * np.interp(wavelength, self.chiar[:, 0], self.chiar[:, 1])
            found = True

        if (not found):
            raise Exception('Extinction law not found')
            	
        return 10.0**(-0.4*extinction_curve)

    def _agn(self, wavelength):

        n = len(wavelength)

        lambdah = 0.01
        lambdau = 0.1
        lambdaRJ = 1.0
        p = 0.5
        const = 0.2784

        agn = np.zeros(n)

        ind = np.where(wavelength <= lambdah)[0]
        agn[ind] = const * wavelength[ind]**1.2 / lambdah**1.2

        ind = np.where((wavelength > lambdah) & (wavelength <= lambdau))[0]
        agn[ind] = const

        ind = np.where((wavelength > lambdau) & (wavelength <= lambdaRJ))[0]
        agn[ind] = const * wavelength**(-p) / lambdau**(-p)

        ind = np.where(wavelength > lambdaRJ)[0]
        agn[ind] = const * lambdaRJ**(-p) / lambdau**(-p) * wavelength**(-3) / lambdaRJ**(-3)

        return agn

    def _parse_prior(self, priors):

        # Read priors
        self.type = {}
        self.limits = {}
        self.rv = {}
        self.central = {}

        if (self.verbose):
            print(f"PRIORS")

        # Add default ranges depending on the range of parameters
        for key in self.parameter_names:
            value = priors[key]
        
            
            # Extract the range of parameters (if given) and check that they
            # are inside the defined interval. If not, fallback to the pre-defined
            # in the database
            tmp = re.findall('range\((.+?)\)', value)
            if (len(tmp) == 0):
                self.limits[key] = self.default_limits[key]
            else:
                self.limits[key] = [float(i) for i in tmp[0].split(',')]
            
            # Uniform priors
            if ('uniform' in value):
                self.type[key] = 'uniform'

                # Extract limits of the uniform prior
                tmp = re.findall('uniform\((.+?)\)', value)
                prior_values = [float(i) for i in tmp[0].split(',')]

                # Check that the limits of the uniform prior are not outside of the range
                if ((prior_values[0] < self.limits[key][0]) | (prior_values[1] > self.limits[key][1])):
                    raise Exception(f'Range for parameters {key} larger than allowed.')

                # Check that the limits are in ascending order                    
                if (prior_values[0] >= prior_values[1]):
                    raise Exception(f'Uniform prior for variable {key} has incorrect range. Lower ({prior_values[0]}) >= Upper ({prior_values[1]}).')

                self.limits[key] = prior_values
                self.rv[key] = uniform(loc=prior_values[0], scale=prior_values[1] - prior_values[0])
                self.central[key] = 0.5 * (prior_values[1] + prior_values[0])

                if (self.verbose):
                    print(f" - {key:12s} - uniform in range [{self.limits[key][0]},{self.limits[key][1]}]")
                
            # Normal prior
            if ('normal' in value):
                self.type[key] = 'normal'

                # Extract limits of the uniform prior
                tmp = re.findall('normal\((.+?)\)', value)
                prior_values = [float(i) for i in tmp[0].split(',')]
                
                # Define the prior properties, using a truncated normal                
                a = (self.limits[key][0] - prior_values[0]) / prior_values[1]
                b = (self.limits[key][1] - prior_values[0]) / prior_values[1]
                self.rv[key] = truncnorm(loc=prior_values[0], scale=prior_values[1], a=a, b=b)
                self.central[key] = prior_values[0]

                if (self.verbose):
                    print(f" - {key:12s} - normal(\u03bc={prior_values[0]},\u03c3={prior_values[1]}) in range [{self.limits[key][0]},{self.limits[key][1]}]")

            # Dirac prior
            if ('dirac' in value):
                self.type[key] = 'dirac'

                # Extract limits of the uniform prior
                tmp = re.findall('dirac\((.+?)\)', value)                
                prior_values = [float(tmp[0]), 0.01*self.typical[key]]
                
                # Define the prior properties, using a truncated normal                
                a = (self.limits[key][0] - prior_values[0]) / prior_values[1]
                b = (self.limits[key][1] - prior_values[0]) / prior_values[1]
                self.rv[key] = truncnorm(loc=prior_values[0], scale=prior_values[1], a=a, b=b)
                self.central[key] = prior_values[0]

                if (self.verbose):
                    print(f" - {key:12s} - dirac(\u03bc={prior_values[0]}) in range [{self.limits[key][0]},{self.limits[key][1]}]")

    def _read_nenkova(self):
        path = str(__file__).split('/')
        root = '/'.join(path[0:-1])
        f = FortranFile(f'{root}/nenkova/compressed_database.bin', 'r')        
        self.nY, self.nn0, self.nq, self.ntauv, self.nsig, self.ni, self.npca, self.nlam = f.read_ints()
        Y = f.read_reals(dtype='float32')
        sig = f.read_reals(dtype='float32')
        n0 = f.read_reals(dtype='float32')
        q = f.read_reals(dtype='float32')
        tauv = f.read_reals(dtype='float32')
        inc = f.read_reals(dtype='float32')

        self.base_wavelength = f.read_reals(dtype='float32')
        self.base = f.read_reals(dtype='float32').reshape((self.npca, self.nlam), order='F')
        self.coefs = f.read_reals(dtype='float32').reshape((self.npca, self.ni, self.ntauv, self.nq, self.nn0, self.nsig, self.nY), order='F')
        self.meanSED = f.read_reals(dtype='float32')
        
        nenkova.init(Y, sig, n0, q, tauv, inc, self.base_wavelength, self.base, self.coefs, self.meanSED)
        
        # self.interpolators = [None] * self.npca
        # for i in range(self.npca):
            # self.interpolators[i] = RegularGridInterpolator((Y, sig, n0, q, tauv, inc), self.coefs[i, ...].T)

        f.close()

        if (self.verbose):
            print("Nenkova database read")

    def _read_hoenig_neural(self):
        
        # Read frequency/wavelength data
        path = str(__file__).split('/')
        root = '/'.join(path[0:-1])
        data_file = np.load(f"{root}/neural/datos_clumpy_parm_x.npz", allow_pickle=True)            
        self.x_freq = data_file['x_freq']
        self.base_wavelength = 10.0**data_file['x_ldo'].astype('float')

        # Read interpolating NN            
        checkpoint = f'{root}/neural/2021-10-19-13:04:24.pth'
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        hyperparameters = chk['hyperparameters']
        self.model = model_nn.Network(hyperparameters).to(self.device)
        self.model.load_state_dict(chk['state_dict'])
        self.model.eval()

        # Read autoencoder
        checkpoint = f'{root}/neural/2021-09-28-17:43:00.pth_encoder'
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        hyperparameters = chk['hyperparameters']
        self.model_encoder = model_encoder.Network(hyperparameters).to(self.device)
        self.model_encoder.load_state_dict(chk['state_dict'])
        self.model_encoder.eval()

        if (self.verbose):
            print("Hoenig model NN read")
        
    def prior_transform_nested(self, u):
        """Transforms samples `u` drawn from the unit cube to samples to those
        from our uniform prior within [-10., 10.) for each variable."""

        x = np.array(u)
        for i, (key, rv) in enumerate(self.rv.items()):
            x[i] = rv.ppf(u[i])
                
        return x

    def logprior_emcee(self, theta):
        logprior = 0.0

        for i, (key, rv) in enumerate(self.rv.items()):        
            logprior += rv.logpdf(theta[i])

        return logprior

    def nenkova_sed(self, x):
        # for i in range(self.npca):
            # self.coefs[i] = self.interpolators[i](x)

        self.coefs = nenkova.synth(x.astype('float32'))

        sed = np.einsum('i,ij->j', self.coefs, self.base)
        sed = 10.0**(sed + self.meanSED)

        return sed

    def hoenig_sed(self, x):
        xin = np.copy(x)
        
        xin[0] = (x[0] - self.default_limits['a'][0]) / (self.default_limits['a'][1] - self.default_limits['a'][0])
        xin[1] = (x[1] - self.default_limits['n0'][0]) / (self.default_limits['n0'][1] - self.default_limits['n0'][0])
        xin[2] = (x[2] - self.default_limits['h'][0]) / (self.default_limits['h'][1] - self.default_limits['h'][0])
        xin[3] = (x[3] - self.default_limits['aw'][0]) / (self.default_limits['aw'][1] - self.default_limits['aw'][0])
        xin[4] = (x[4] - self.default_limits['theta_w'][0]) / (self.default_limits['theta_w'][1] - self.default_limits['theta_w'][0])
        xin[5] = (x[5] - self.default_limits['theta_sig'][0]) / (self.default_limits['theta_sig'][1] - self.default_limits['theta_sig'][0])
        xin[6] = (x[6] - self.default_limits['f_wd'][0]) / (self.default_limits['f_wd'][1] - self.default_limits['f_wd'][0])
        xin[7] = (x[7] - self.default_limits['rout'][0]) / (self.default_limits['rout'][1] - self.default_limits['rout'][0])
        xin[8] = (x[8] - self.default_limits['angle'][0]) / (self.default_limits['angle'][1] - self.default_limits['angle'][0])

        inp_tau_cl = 1.
        inp_dist = 0.
        
        # Order of inputs when training
        inp_values = np.array((xin[0],xin[1],xin[2],xin[3],xin[4],xin[5],xin[6],xin[7],inp_tau_cl,inp_dist,xin[8]))

        with torch.no_grad():        
            # We transform the input from Numpy to PyTorch tensor
            inputs = torch.tensor(inp_values.astype('float32')).to(self.device)
            out_nn = self.model(inputs)        

        # Format to apply decoder (and recover the parameter stratification)
        out_nn_t = out_nn.unsqueeze(1).unsqueeze(0)

        # Applying the decoder on the predicted values by the NN
        # We only need the forward pass, so we do not accumulate gradients
        with torch.no_grad():            
            inputs = out_nn_t.to(self.device)
            out_de = self.model_encoder.decoder(inputs)

        # Convert to array
        # We bring the result to the CPU (if it was on the GPU) and transform to Numpy
        out_de_n = out_de.cpu().numpy()

        out = out_de_n[0, 0, :] * (self.max_n - self.min_n) + self.min_n

        # I still need to find where this 1e6 is coming from. It is needed for a proper comparison with Nenkova models
        return 10.0**out / 1e6

    def loglike(self, x, only_synth=False):
        
        # Given in lambda*F_lambda (equivalent to nu*F_nu)
        if (self.model_type == 'nenkova'):        
            sed = self.nenkova_sed(x[0:6])

        # Given in nu*F_nu (equivalent to lambda*F_lambda)
        if (self.model_type == 'hoenig'):
            sed = self.hoenig_sed(x[0:9])

        # y : [5,100]
        # sigma: [15,70]
        # n: [1,15]
        # q: [0,3]
        # tauv:[10, 300]
        # i: [0,90]        

        # Add AGN if desired
        if (self.use_agn):
            sed += self._agn(self.base_wavelength)
        
        # Tranform to Jansky
        sed *= (self.base_wavelength * 1e-4 / 2.99792458e10) / 1e-26

        # Apply extinction curve
        sed *= self._extiction_curve(self.extiction_law, self.base_wavelength, x[-2])

        # Normalize to 1e10 and apply shift
        sed *= 10.0**x[-3] / 1e10
        
        chi2 = 0.0

        if (only_synth):
            sed = np.interp(self.base_wavelength, self.base_wavelength * (1.0 + x[-1]), sed)
            return sed

        # Now synthesize the flux in the photometric filters
        for i in range(self.obs.n_filters):

            # Interpolate the synthetic SED to the wavelength in the filter and
            # apply the redshift locally to the filter
            sed_filter = np.interp(self.obs.transmission[i][0, :], self.base_wavelength * (1.0 + x[-1]), sed)

            # Compute flux in the filter (the filter is already normalized to unit area)
            syn_flux = np.trapz(sed_filter * self.obs.transmission[i][1, :], self.obs.transmission[i][0, :])
        
            chi2 += (self.obs.obs_flux[i] - syn_flux)**2 / self.obs.obs_sigma[i]**2
        
        # Now take into account spectroscopy
        if (self.obs.n_spec != 0):

            # Interpolate the synthetic SED to the wavelength in the spectroscopic range and
            # apply the redshift locally to the filter            
            sed_filter = np.interp(self.obs.obs_wave[-1], self.base_wavelength * (1.0 + x[-1]), sed)

            chi2 += np.sum( (self.obs.obs_flux[-1] - sed_filter)**2 / self.obs.obs_sigma[-1]**2)

        loglike = -0.5 * chi2

        return loglike
        
    def loglike_emcee(self, x):
        log_prior = self.logprior_emcee(x)
        loglike = self.loglike(x)

        return log_prior + loglike
