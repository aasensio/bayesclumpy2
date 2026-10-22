import numpy as np
import matplotlib.pyplot as pl


def agn(wavelength, const=0.2784):
    n = len(wavelength)

    lambdah = 0.03
    lambdau = 0.3
    lambdaRJ = 3.0
    p = 4.0/3.0    
    
    agn = np.zeros(n)
    ind = np.where(wavelength <= lambdah)[0]    
    agn[ind] = const * wavelength[ind] / lambdah

    ind = np.where((wavelength > lambdah) & (wavelength <= lambdau))[0]    
    agn[ind] = const

    ind = np.where((wavelength > lambdau) & (wavelength <= lambdaRJ))[0]
    agn[ind] = const * wavelength[ind]**(-p) / lambdau**(-p)
    
    ind = np.where(wavelength > lambdaRJ)[0]    
    agn[ind] = const * lambdaRJ**(-p) / lambdau**(-p) * wavelength[ind]**(-3) / lambdaRJ**(-3)
    return agn

if __name__ == '__main__':
        rsub = 0.288 * 3.086e16
        factor = 4.0 * np.pi * rsub**2 * 1e7
        tmp = np.loadtxt('AGN_SPECTRUM_tornorm.txt')

        wavelength = np.logspace(-8, 8, 100)

        nu = 3e14 / wavelength

        nu_fnu = agn(wavelength, const = 3e5*0.813)
        f_nu = nu_fnu / nu

        area = factor * np.abs(np.trapezoid(f_nu, x=nu))

        area_seb = factor * np.abs(np.trapezoid(tmp[:, 2] / tmp[:, 1], x=tmp[:, 1]))
        print(f'Area ours (cgs) : {area} - {area/1e46}:')
        print(f'Area SEB (cgs)  : {area_seb} - {area_seb/1e46}:')

        # nu_fnu = agn(wavelength, const = area)

        pl.loglog(nu, nu_fnu)
        pl.loglog(tmp[:, 1], tmp[:, 2])
        pl.xlabel('Frequency (Hz)')
        pl.ylabel('AGN')
        pl.title('AGN SED')
        pl.grid(True)
        pl.show()