import jax 
import jax.numpy as jnp 
from jax import vmap 
from jax.scipy.optimize import minimize
from scipy import stats 
import numpy as np 
import time 
from Vares import Auxiliary
from functools import partial
from scipy import optimize
import arch
from arch import arch_model

jax.config.update("jax_enable_x64", True)

# TODO
# 1. Variance targeting
# 2. Provide rescaling (??)

def nll(returns, parameters): 
    '''GARCH(1,1) negative Likelyhood'''
    alpha, beta, omega, mu = parameters
    T = len(returns) 
    sigma2 = jnp.zeros(T)
    sigma2 = sigma2.at[0].set(jnp.var(returns))

    def update_sigma2(sigma2_prev, t): 
        innovation = (returns[t-1] - mu)**2 
        sigma2_new = omega + alpha * innovation + beta * sigma2_prev

        # First: carry to next iteration
        # Second: accumulate in output array
        return sigma2_new, sigma2_new

    _, sigma2_rest = jax.lax.scan(update_sigma2, sigma2[0], jnp.arange(1, T))
    sigma2 = sigma2.at[1:].set(sigma2_rest)

    log_likelyhood_t = jnp.log(sigma2) + ((returns - mu)**2 / sigma2)
    NLL = jnp.sum(log_likelyhood_t)

    return NLL 

@Auxiliary.timer
@jax.jit
def ManMLE_garch(returns, parameters):
    alpha, beta, omega, mu = parameters 

    #ensuring non-neegativity of the variance parameres
    variance_greeks = jnp.array([alpha, beta, omega])
    log_alpha, log_beta, log_omega = jnp.log(variance_greeks)

    opt_params = jnp.array([log_alpha, log_beta, log_omega, mu])

    def objective_function(optimization_parameters, returns): 
        log_alpha, log_beta, log_omega, mu = optimization_parameters
        log_variance_greeks = jnp.array([log_alpha, log_beta, log_omega])
        alpha, beta, omega = jnp.exp(log_variance_greeks)
        ll_parameters = jnp.array([alpha, beta, omega, mu])
        return nll(returns, ll_parameters)
        
    result = minimize(objective_function, opt_params, args=(returns, ), method='BFGS')

    return result 

def simulation(mean, std, size): 
    _ = stats.norm.rvs(loc=mean, scale=std, size=size)
    return _
    
def MLE_params_init():
    pass

def MLE_garch(returns):
    pass

def simulate_garch_path(alpha, beta, omega, mu, T, sigma2_0 = None): 
    '''Uses the following notation: 
        r_t = mu  + eps_t 
        eps_t = sigma_t * z_t, where z_t is distributed N(0,1)
        sigma2_t = omega + alpha * eps_t-1 + beta * sigma2_t-1
    '''
    sigma2 = np.zeros(T) 
    z = stats.norm.rvs(size=T)
    r = np.zeros(T)
    eps = np.zeros(T)

    sigma2[0] = omega / (1 - alpha - beta) if sigma2_0 is None else sigma2_0
    eps[0] = sigma2[0]**(1/2) * z[0]

    for t in range(1, T): 
        sigma2[t] = omega + alpha * eps[t-1] + beta * sigma2[t-1]
        eps[t] = sigma2[t] * z[t]

    r = eps + mu 

    return r 

if __name__ == '__main__': 
    mean, std, size = 1, 0.2, 5000 

    # returns = simulation(mean, std, size)

    # returns = simulate_garch_path(0.1, 0.8, 0.5, 0.1, 5000)
    returns = simulate_garch_path(0.1, 0.8, 0.03, 0.1, 5000)


    mu = jnp.mean(returns)
    alpha = 0.05
    beta = 0.9
    omega = jnp.var(returns) * (1 - alpha - beta)

    parameters = jnp.array([alpha, beta, omega, mu])

    results = ManMLE_garch(returns, parameters)

    print(jnp.exp(results.x[:3]), results.x[4])

    print(results.status)

    # fit a GARCH(1,1) with constant mean
    am = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off')

    # print(res.summary())
    print("params:", res.params)
