import numpy as np
from scipy.special import gammaln
from scipy.stats import t as tdist

def negloglike_clayton(alpha, u):
    powu = u**-alpha
    lnu = np.log(u)
    logC = (-1./alpha)*np.log(np.sum(powu, axis=1) - 1)
    logy = np.log(alpha+1) + (2.*alpha+1.)*logC - (alpha+1.)*np.sum(lnu, axis=1)
    nll = -np.sum(logy)
    
    if nargout > 1:
        dlogy = 1./(1+alpha) - logC/alpha + (2+1./alpha)*np.sum(powu*lnu, axis=1)**2 / (np.sum(powu, axis=1) - 1)  - np.sum(lnu, axis=1)
        d2 = np.sum(dlogy**2)
        return nll, d2
    
    return nll

def negloglike_frank(alpha, u):
    expau = np.exp(alpha * u)
    sumu = np.sum(u, axis=1)
    if np.abs(alpha) < 1e-5:
        logy = 2*alpha*np.prod(u-.5, axis=1) 
    else:
        logy = np.log(-alpha*np.expm1(-alpha)) + alpha*sumu - 2*np.log(np.abs(1 + np.exp(alpha*(sumu - 1)) - np.sum(expau, axis=1)))
    nll = -np.sum(logy)
    
    if nargout > 1:
        if np.abs(alpha) < 1e-5:
            dlogy = 2*np.prod(u-.5, axis=1)
        else:
            dlogy = 1./alpha + 1./np.expm1(alpha) + sumu - 2*((sumu-1)*np.exp(alpha*(sumu-1)) - np.sum(u*expau, axis=1)) / (1 + np.exp(alpha*(sumu-1)) - np.sum(expau, axis=1))
        d2 = np.sum(dlogy**2)
        return nll, d2
    
    return nll

def negloglike_gs(rho, u):
    x = scipy.stats.norm.ppf(u[:,0])
    y = scipy.stats.norm.ppf(u[:,1])
    F = -np.sum(np.log( (1-rho**2)**(-.5) * np.exp((x**2 + y**2)/2 + (2*rho*x*y - x**2 - y**2)/(2*(1-rho**2))) ))
    return F

def negloglike_gumbel(alpha, u):
    v = -np.log(u)
    v = np.sort(v, axis=1)
    vmin = v[:,0]
    vmax = v[:,1]
    logv = np.log(v)
    nlogC = vmax*(1 + (vmin/vmax)**alpha)**(1./alpha)
    lognlogC = np.log(nlogC)
    logy = np.log(alpha - 1 + nlogC) - nlogC + np.sum((alpha-1)*logv + v, axis=1) + (1-2*alpha)*lognlogC
    nll = -np.sum(logy)
    
    if nargout > 1:
        dnlogC = nlogC *(-lognlogC + np.sum(logv*v**alpha, axis=1)/np.sum(v**alpha, axis=1)) / alpha
        dlogy = (1+dnlogC)/(alpha-1+nlogC) - dnlogC + np.sum(logv, axis=1) + (1-2*alpha)*dnlogC/nlogC - 2*lognlogC
        d2 = np.sum(dlogy**2)
        return nll, d2
    
    return nll
    
def negloglike_t(nu, R, u):
    t = tdist.ppf(u, df=nu)
    n,d = t.shape
    R = R / np.sqrt(np.sum(R**2, axis=1))[:,None]
    tRinv = t / R
    
    nll = - n*gammaln((nu+d)/2) + n*d*gammaln((nu+1)/2) - n*(d-1)*gammaln(nu/2) + n*np.sum(np.log(np.abs(np.diag(R)))) + ((nu+d)/2)*np.sum(np.log(1 + np.sum(tRinv**2, axis=1)/nu)) - ((nu+1)/2)*np.sum(np.log(1 + t**2/nu))
    return nll
