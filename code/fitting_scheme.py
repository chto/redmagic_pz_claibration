from scipy.stats import multivariate_normal
import numpy as np
import sys
def run_em(x, w, phi, mu, sigma, silent=False):
    """
    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-1  # Convergence threshold
    max_iter = 100

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        m, n = x.shape
        m, k = w.shape
        #E step:
        if ll is None:
            for i in range(k):
                p_given_z = np.array([multivariate_normal.pdf(x[j], mu[i], sigma[i]) for j in range(m)]) 
                w[:, i] = phi[i]*p_given_z
            norm = np.sum(w, axis=1).reshape(-1,1)
            w = w/norm 
        else:
            w = w
        #M step :
        phi = np.mean(w, axis=0) 
        mu = w.T.dot(x)/np.sum(w, axis=0).reshape(-1,1)
        assert (mu.shape==(k,n))
        for i in range(k):
            sigma[i] = np.dot((w[:,i].reshape(-1,1)*(x-mu[i])).T, (x-mu[i]))/np.sum(w[:,i])+1E-9*np.identity(n)
        assert (sigma[0].shape == (n,n))

        for i in range(k):
            p_given_z = np.array([multivariate_normal.pdf(x[j], mu[i], sigma[i]) for j in range(m)]) 
            w[:, i] = phi[i]*p_given_z
        norm = np.sum(w, axis=1).reshape(-1,1)
        w = w/norm 
        prev_ll = ll
        ll = np.sum(np.log(norm))
        if not silent:
            print("iteration, log likelihood", it, ll)
            sys.stdout.flush()
        if prev_ll is not None:
            assert (ll>=prev_ll)
        it += 1
    f1 = np.mean(w,axis=0)
    return w, f1,  mu.squeeze(),np.array(sigma).squeeze()

def fit_function(x, param):
    f_in, mu, cov, ngaussian = param
    y = 0
    for i in range(ngaussian):
        y += f_in[i]*multivariate_normal.pdf(x, mu[i], cov[i])
    return y

class EM_GMM():
  def __init__(self, ngaussian):
    self.ngaussian = ngaussian
    self.silent=True
    self.w = None
    self.phi = None
    self.mu = None
    self.cov = None
    self.f = None
    self.param = (self.f, self.mu, self.cov, self.ngaussian)
  def fit(self, x):
    self.w = np.ones((x.shape[0],self. ngaussian))/float(self.ngaussian)
    self.phi = np.ones(self.ngaussian)
    self.mu = []
    for _ in range(self.ngaussian):
        self.mu.append([np.random.uniform(x[:,i].min(), x[:,i].max()) for i in range(x.shape[1])])
    self.cov = [np.cov(x, rowvar=False) for _ in range(self.ngaussian)]
    self.w, self.f, self.mu, self.cov = run_em(x, self.w, self.phi, self.mu, self.cov, silent=self.silent)
    self.param = (self.f, self.mu, self.cov, self.ngaussian)
