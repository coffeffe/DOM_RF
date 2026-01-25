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

#Учитывая отсутствующие котировки опционов, получившееся распределение можно и в поверхность переводить

# TODO 14/10/2025
# 1. Variance targeting - DONE 
# 2. Provide rescaling (??) - DONE 
# 3. correct initial parameters initialization - DONE 
# 4. Check out arch packages for their methods

# TODO
# 1. Non-Gaussian MLE estimation + Quasi MLE (consult arch/base/distributions.py)
# 2. Introduce proper boundary values - w/o jnp.log etc. (together with persistence)
# 3. Consider NOT estimating mean parameter
# 4. persistence checks (PRIORYTY) - consider: kappa = 1 - alpha - beta 
# 5. Test MLE w/ & w/o bounds. (naive)
# 6. penalty MLE
# 7. !!! GARCH quality metric (PRIORITY) 
# 8. Design test for rescaled returns against non-resclaed
# 9. Grid search for init parameters


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
    '''Manual Maximum Likelyhood Estimation for the GARCH parameters. 
    The initial guess (parameters before estimation) have to be prespecified
    Here logarithm transformation is applied for alpha, beta and omega parameters (for bounding on [0;+inf])'''
    alpha, beta, omega, mu = parameters 

    #ensuring non-negativity of the variance parameres
    variance_greeks = jnp.array([alpha, beta, omega])
    log_alpha, log_beta, log_omega = jnp.log(variance_greeks)

    opt_params = jnp.array([log_alpha, log_beta, log_omega, mu])

    def objective_function(optimization_parameters, returns):
        '''input parameters are transformed into their original version'''
        log_alpha, log_beta, log_omega, mu = optimization_parameters
        log_variance_greeks = jnp.array([log_alpha, log_beta, log_omega])
        alpha, beta, omega = jnp.exp(log_variance_greeks)
        ll_parameters = jnp.array([alpha, beta, omega, mu])
        return nll(returns, ll_parameters)
        
    result = minimize(objective_function, opt_params, args=(returns, ), method='BFGS')

    return result 

@jax.jit 
def ManMLE_garch_naive(returns, parameters): 
    '''Errors may occure in optimization as parameters: jax.Array are optimized as well as input.'''
    def objective_function(optimization_parameters, returns): 
        return nll(returns, optimization_parameters)
        
    result = minimize(objective_function, parameters, args=(returns, ), method='BFGS')

    return result 

@Auxiliary.timer
@jax.jit
def VTE_garch(returns, parameters): 
    '''Variance Targerting Estimation of GARCH parameters. 
    For details on the method consult https://mpra.ub.uni-muenchen.de/15143/1/MPRA_paper_15143.pdf. 
    The Script presented here is the JAX realization of the method described in the paper. 
    Here logarithm transformation is applied for alpha and beta parameters (for bounding on [0;+inf]) 
    !!! TRY logarithm bounding kappa (1- alpha - beta)
    !!! Handle the persistence condition 
    '''
    #checks for persistence are necessary
    alpha, beta, mu = parameters 
    residuals = (returns - jnp.mean(returns))**2 
    gamma = jnp.var(residuals) # one can move these calculations into objevtive_function()
    # omega = gamma * kappa '
    
    #ensuring non-negativity of the variance parameres
    variance_greeks = jnp.array([alpha, beta])
    log_alpha, log_beta= jnp.log(variance_greeks) 

    opt_params = jnp.array([log_alpha, log_beta, mu])

    def objective_function(optimization_parameters, returns, gamma):
        '''input parameters are transformed into their original version'''
        log_alpha, log_beta, mu = optimization_parameters
        log_variance_greeks = jnp.array([log_alpha, log_beta])
        alpha, beta = jnp.exp(log_variance_greeks)
        
        #omega handling 
        kappa = 1 - alpha - beta 
        omega = gamma * kappa 

        ll_parameters = jnp.array([alpha, beta, omega, mu])
        return nll(returns, ll_parameters)

    result = minimize(objective_function, opt_params, args=(returns, gamma), method='BFGS')

    return result 

def VTE_garch_naive(returns, parameters):
    
    def objective_function(optimization_parameters, returns): 
        alpha, beta, mu = optimization_parameters

        residuals = (returns - jnp.mean(returns))**2 
        gamma = jnp.var(residuals) # one can move these calculations into objevtive_function()
        kappa = 1 - alpha - beta
        omega = gamma * kappa 

        ll_parameters = jnp.array([alpha, beta, omega, mu])
        return nll(returns, ll_parameters)
    
    result = minimize(objective_function, parameters, args=(returns, gamma), method='BFGS') 

    return result   

def initialize_parameters_test(returns): 
    alpha = 0.1 
    beta = 0.85 
    mu = np.mean(returns)
    
    residuals = (returns - mu)**2 
    gamma = np.var(residuals)

    kappa = 1 - alpha - beta 
    omega = kappa * gamma 

    return {
        'alpha': alpha,
        'beta': beta,
        'mu': mu,
        'gamma': gamma, 
        'kappa': kappa,
        'omega': omega
    }

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

    parameters_MMLE = jnp.array([alpha, beta, omega, mu])

    results_MMLE = ManMLE_garch(returns, parameters_MMLE)

    print(jnp.exp(results_MMLE.x[:3]), results_MMLE.x[4])

    print(results_MMLE.status)

    print('+++++++++++++++++++++++++++++++++')

    returns_rescaled = 100 * returns 

    results_MMLE_rescaled = ManMLE_garch(returns_rescaled, parameters_MMLE)

    print(jnp.exp(results_MMLE_rescaled.x[:3]), results_MMLE_rescaled.x[4])

    print(results_MMLE_rescaled.status)

    print('+++++++++++++++++++++++++++++++++')

    parameters_VTE = jnp.array([alpha, beta, mu])

    results_VTE = VTE_garch(returns, parameters_VTE)

    print(jnp.exp(results_VTE.x[:2]), results_VTE.x[3])

    print(results_VTE.status)

    # # fit a GARCH(1,1) with constant mean
    # am = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
    # res = am.fit(disp='off')

    # # print(res.summary())
    # print("params:", res.params)
