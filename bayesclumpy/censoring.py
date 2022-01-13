import numpy as np
import scipy.special as sp
from scipy.stats import norm
import matplotlib.pyplot as pl

# Gaussian Process Regression with Censored Data Using
# Expectation Propagation
# Perry Groot, Peter Lucas

l = 1.0
s = 0.2
n = 2000
f = np.linspace(0.0, 2.0, n+1)
sqrt2 = np.sqrt(2.0)
sqrtpi = np.sqrt(np.pi)

cdf_integral_all = 0.5 * sp.erf(1.0/(sqrt2 * s)) + 0.5 + s / (sqrt2 * sqrtpi) * np.exp(-1.0/(2*s**2))

cdf_integral = l * sp.erf(l/(sqrt2 * s)) + s / (sqrt2 * sqrtpi) * (np.exp(-l**2/(2*s**2)) - 1.0)

print(cdf_integral, cdf_integral_all, cdf_integral / cdf_integral_all)

fig, ax = pl.subplots()

cdf = norm.cdf((l - f) / s)
ax.plot(f, cdf)

print(np.trapz(cdf, f))
print(np.trapz(cdf[0:n//2], f[0:n//2]) / np.trapz(cdf, f))