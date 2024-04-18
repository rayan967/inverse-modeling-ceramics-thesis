import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from jax import random, jit, value_and_grad

def kernel(X1, X2, theta):
    """
    Anisotropic squared exponential kernel.
    Every dimension gets scales by l_i , such that 
    we have dim+1 kernel parameters (+1 is sigma.)
    
    Args:
        X1: Array of m points (m, d).
        X2: Array of n points (n, d).
        theta: kernel parameters (1+dim,)
    """
    N1 = X1.shape[0]
    D1 = X1.shape[1]

    N2 = X2.shape[0]
    D2 = X2.shape[1]
    
    dim = X1.shape[1]
    
    L     = theta[0:]
   

    # Prescale data
    X1 = X1/L
    X2 = X2/L

    n1sq = jnp.sum(X1**2,axis=1);
    n2sq = jnp.sum(X2**2,axis=1);

    DXY = jnp.transpose(jnp.outer(jnp.ones(N2),n1sq)) + jnp.outer(jnp.ones(N1),n2sq)-2* (jnp.dot(X1,jnp.transpose(X2)))
    DXY[np.abs(DXY) < 1E-8] = 0
    
    kxy = jnp.exp(-DXY / 2.0)
    
    return kxy

def jitter(d, value=1e-4):
    return jnp.eye(d) * value


def softplus(X):
    return jnp.log(1 + jnp.exp(X))


def softplus_inv(X):
    return jnp.log(jnp.exp(X) - 1)


def identity(X):
    return X


def identity_inv(X):
    return X


def pack_params(theta, X_m):
    return jnp.concatenate([identity_inv(theta), X_m.ravel()])


def unpack_params(params):
    return identity(params[:2]), jnp.array(params[2:].reshape(-1, 1))

def nlb_fn(X, y, epsilon):
    n = X.shape[0]

    def nlb(params):
        """
        Negative lower bound on log marginal likelihood.

        Args:
            params: kernel parameters `theta` and inducing inputs `X_m`
        """

        theta, X_m = unpack_params(params)
        
        X_m = X_m.reshape((-1,2))
        
        K_mm = kernel(X_m, X_m, theta) + jitter(X_m.shape[0]) 
        K_mm_inv = jnp.linalg.inv(K_mm)
        
        K_nm = kernel(X, X_m, theta)
        K_nn = kernel(X, X, theta)
        
        E = jnp.diagflat(epsilon**2)
        Q_nn = K_nm@K_mm_inv@K_nm.T

        Lambda     = E + jnp.diagflat(jnp.diagonal(K_nn - Q_nn))
        Lambda_inv = jnp.linalg.inv(Lambda)

        # Cholesky decomp. of Lambda + Qnn
        LS = jnp.linalg.cholesky(Lambda+Q_nn)
        c1 = jsp.linalg.solve_triangular(LS, y, lower=True) # m x 1
        c2 = jsp.linalg.solve_triangular(LS.T, c1, lower=False) # m x 1

        lb  = -n/2 * jnp.log(2 * jnp.pi)
        lb -= jnp.sum(jnp.log(jnp.diagonal(LS)))
        lb -= 0.5*y.T@c2
        lb -= 0.5*jnp.trace(Lambda_inv@(K_nn-Q_nn)) # Trace regularization

        return -lb

    # nlb_grad returns the negative lower bound and
    # its gradient w.r.t. params i.e. theta and X_m.
    nlb_grad = jit(value_and_grad(nlb))

    def nlb_grad_wrapper(params):
        value, grads = nlb_grad(params)
        # scipy.optimize.minimize cannot handle
        # JAX DeviceArray directly. a conversion
        # to Numpy ndarray is needed.
        return jnp.array(value), jnp.array(grads)

    return nlb_grad_wrapper

@jit
def phi_opt(theta, X_m, X, y, epsilon):
    
    E = jnp.diagflat(epsilon**2)

    K_mm = kernel(X_m, X_m, theta) + jitter(X_m.shape[0])   
    K_mm_inv = jnp.linalg.inv(K_mm)

    K_nm = kernel(X, X_m, theta)
    K_mn = K_nm.T

    K_nn = kernel(X, X, theta)
    
    Q_nn =  K_nm@K_mm_inv@K_nm.T

    Lambda     = E + jnp.diagflat(jnp.diagonal(K_nn - Q_nn))
    Lambda_inv = jnp.linalg.inv(Lambda)

    Sigma     = K_mm + K_mn @ Lambda_inv @ K_nm
    Sigma_inv = jnp.linalg.inv(Sigma)

    mu_m = (K_mm @ Sigma_inv @ K_mn@ Lambda_inv).dot(y)
    #mu_m = (K_mm @ Sigma_inv @ K_mn).dot(y)
    A_m  = K_mm @ Sigma_inv @ K_mm

    return mu_m, A_m, K_mm_inv

@jit
def q(X_test, theta, X_m, mu_m, A_m, K_mm_inv):
    """
    Approximate posterior. 

    Computes mean and covariance of latent 
    function values at test inputs X_test.
    """

    K_ss = kernel(X_test, X_test, theta)
    K_sm = kernel(X_test, X_m, theta)
    K_ms = K_sm.T
    
    
    # Calculate mean 
    f_q = (K_sm @ K_mm_inv).dot(mu_m)
    
    # Calculate covariance 
    f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + \
        K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms
        
    # Calculate derivative    
    #df_q = (dK_sm @ K_mm_inv).dot(mu_m)

    return f_q, f_q_cov#, df_q