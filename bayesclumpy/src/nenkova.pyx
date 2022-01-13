# cython: language_level=3
from numpy cimport ndarray as ar
from numpy import empty, ascontiguousarray

# STUFF = "Hi"

cdef extern:
	void c_initialize_database(int *nY, int *nsigma, int *nn0, int *nq, int *ntauv, int *ni, int *npca, int *nlam, float *Y, float *sigma, 
		float *n0, float *q, float *tauv, float *i, float *lam, float *base, float *coefs, float *meanSED)

	void c_lininterpol_database(float *pars, float *coefs)
	
def init(ar[float,ndim=1] Y, ar[float,ndim=1] sigma, ar[float,ndim=1] n0, ar[float,ndim=1] q, ar[float,ndim=1] tauv, ar[float,ndim=1] i, 
	ar[float,ndim=1] lam, ar[float,ndim=2] base, ar[float,ndim=7,mode='fortran'] coefs, ar[float,ndim=1] meanSED):

	"""
	Initialization
	"""
	cdef:
		int nY = len(Y)
		int nsigma = len(sigma)
		int nn0 = len(n0)
		int nq = len(q)
		int ntauv = len(tauv)
		int ni = len(i)
		int nlam = len(lam)
		int npca = base.shape[0]
		
	c_initialize_database(&nY, &nsigma, &nn0, &nq, &ntauv, &ni, &npca, &nlam, &Y[0], &sigma[0], &n0[0], &q[0], &tauv[0], &i[0], &lam[0], &base[0,0], &coefs[0,0,0,0,0,0,0], &meanSED[0])
	
	return

def synth(ar[float,ndim=1] pars):

	"""
	evaluate model
	"""
	cdef:
		ar[float,ndim=1] coefs = empty(20, order='F', dtype='float32')
		
	c_lininterpol_database(&pars[0], <float*> coefs.data)
	
	return coefs