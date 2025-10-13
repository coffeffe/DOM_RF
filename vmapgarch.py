import jax 
import jax.numpy as jnp 
from jax import vmap 
from jax.scipy.optimize import minimize
from scipy import stats 
import numpy as np 
import time 
from Vares import Auxiliary
from functools import partial

jax.config.update("jax_enable_x64", True)

#assume constant mean GARCH(p, q) process 
def neg_ll_1_1(params, returns): 
    alpha, beta, omega, mu = params 
    T = len(returns)
    sigma2 = jnp.zeros(T)

    sigma2 = sigma2.at[0].set(omega / (1 - alpha - beta)) #assuming stationary model 

    # @partial(jax.jit, static_argnums=(0,))
    def sigma2_t(t, sigma2): 
        value = omega + alpha * (returns[t-1] - mu)**2 + beta * sigma2[t-1]
        # TODO contral for cases sigma2[t] < 0 if any
        return sigma2.at[t].set(value)
    
    sigma2 = jax.lax.fori_loop(1, T, sigma2_t, sigma2)

    _ = jnp.log(sigma2) + ((returns - mu)**2 / sigma2) 
    neg_ll = jnp.sum(_)    #+ T * jnp.log(2 * jnp.pi)

    return neg_ll

def neg_ll_np(params, retunrs): 
    alpha, beta, omega, mu = params 
    T = len(returns)
    sigma2 = np.zeros(T)

    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, T): 
        value = omega + alpha * (returns[t-1] - mu)**2 + beta * sigma2[t-1]
        sigma2[t] = value 
    
    _ = np.log(sigma2) + ((returns - mu)**2 / sigma2) 
    neg_ll = np.sum(_)
    return neg_ll


def _consraint_perstistance(params): 
    alpha, beta, _, _ = params 
    return 1 - alpha - beta 

@Auxiliary.timer
@jax.jit
def MLE_garch(returns, initial_guess): 
    
    def objective_function(initial_guess, returns):
        return neg_ll_1_1(initial_guess, returns)
        
    result = minimize(objective_function, initial_guess, args=(returns, ), method='BFGS')

    return result
    

if __name__ == '__main__': 
    LOC, SCALE, SIZE = 0.7, 0.7, 50000
    _ = stats.norm.rvs(loc=LOC, scale=SCALE, size=SIZE)
    returns = jnp.array(_)

    initial_guess = jnp.array([0.01, 0.9, 0.4459, 0.7])

    result = MLE_garch(returns, initial_guess)

    print(result.x)
    print(result.status)
    alpha, beta, omega, mu = result.x
    sigma2 = omega + alpha*(returns[-1] - mu) + beta