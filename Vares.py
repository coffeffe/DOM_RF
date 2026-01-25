#idea is all these methods will be unified in the class in the other .py file

#TODO
# 1. Multiple asset VaR (with garch for every asset DCCgarch) - IN WORK 
# 2. VaR for interest rate - IN WORK   
# 3. raise something for the case if xi is close to 0 in case of EVT 
# 4. VaR correlated returns (for the window functions)
# 5. ES (ensure it is optimal and uses the same resources as VaR)
# 6. Write somewhere the fact that EVT performs best for student T with v in [1-5] df 

import jax
from jax import random
from scipy.stats import *
from scipy import stats 
from typing import Union, List, Literal, TypeAlias
import numpy as np
import numpy.typing as npt
import pandas as pd
from functools import wraps, partial
from arch import arch_model
import time
from dataclasses import dataclass
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import exp, pi, sqrt

import jax.numpy as jnp

jax.config.update("jax_enable_x64", True) #for better numerical stability 

number = Union[int, float]
number_like = Union[List[number], number]
array_like = Union[List[number], np.ndarray]
distributions: TypeAlias = Literal["chauchy", "chi2", "expon", "exponpow", "gamma", "lognorm", "norm", "powerlaw", "rayleigh",
                            "uniform", "t", "gumbel_r", "f"]  
FloatArray = npt.NDArray[np.float64]
Floats = Union[float, FloatArray]
Int = Union[int, np.int16, np.int32, np.int64, jnp.int64, jnp.int32, jnp.int16]

ENABLE_TIMING = True  

class Auxiliary:
    
    #might be usefull later
    __DISTRIBUTIONS__ = {
        'cauchy': stats.cauchy,
        'chi2': stats.chi2,
        'expon': stats.expon,
        'exponpow': stats.exponpow,
        'gamma': stats.gamma,
        'lognorm': stats.lognorm,
        'norm': stats.norm,
        'powerlaw': stats.powerlaw,
        'rayleigh': stats.rayleigh,
        'uniform': stats.uniform,
        't': stats.t,
        "gumbel_r": stats.gumbel_r,
        "f": stats.f
    }

    def timer(func): 
            ''' Decorator function for timing function calling. '''
            @wraps(func)
            def wrapper(*args, **kwargs):

                if not ENABLE_TIMING:
                    return func(*args, **kwargs)
                start = time.perf_counter()
                value = func(*args, **kwargs)
                total = time.perf_counter() -start
                print(f'{func.__name__}() took {total:.6f}s')
                return value 
            return wrapper

    def validate_portfolio_inputs(func, summary: bool = False):
        """
        Decorator to validate and convert portfolio returns to 1D jnp.Array format.
        Handles conversions from pandas DataFrame/Series, numpy arrays, and JAX arrays.
        Raises ValueError if input is not 1D after conversion.
        
        The decorated function must accept returns as its first positional argument.
        """
        @wraps(func)
        def wrapper(returns, *args, **kwargs):
            # Check if returns exists
            if returns is None:
                raise ValueError("Portfolio returns data is None. Please provide valid data.")
            
            # Convert pandas DataFrame to jnp.Array
            if isinstance(returns, pd.DataFrame):
                if returns.empty:
                    raise ValueError("Portfolio returns DataFrame is empty. Please check the data source.")
                
                # Check dimensionality before conversion
                if returns.shape[1] > 1:
                    raise ValueError(
                        f"Portfolio returns DataFrame has {returns.shape[1]} columns. "
                        "Expected 1D data. Please compute portfolio returns (e.g., weighted average) "
                        "before passing to this function."
                    )
                
                # Convert single column DataFrame to 1D jnp.Array
                returns = jnp.array(returns.squeeze().values)
            
            # Convert pandas Series to jnp.Array
            elif isinstance(returns, pd.Series):
                returns = jnp.array(returns.values)
            
            # Convert numpy array to jnp.Array
            elif isinstance(returns, np.ndarray):
                if returns.size == 0:
                    raise ValueError("Portfolio returns numpy array is empty.")
                
                # Check dimensionality
                if returns.ndim > 1:
                    raise ValueError(
                        f"Portfolio returns numpy array has {returns.ndim} dimensions "
                        f"with shape {returns.shape}. Expected 1D array. "
                        "Please compute portfolio returns before passing to this function."
                    )
                
                returns = jnp.array(returns)
            
            # Handle JAX arrays (using jax.Array for modern JAX versions)
            elif isinstance(returns, jax.Array):
                if returns.size == 0:
                    raise ValueError("Portfolio returns JAX array is empty.")
                
                # Check dimensionality
                if returns.ndim > 1:
                    raise ValueError(
                        f"Portfolio returns JAX array has {returns.ndim} dimensions "
                        f"with shape {returns.shape}. Expected 1D array. "
                        "Please compute portfolio returns before passing to this function."
                    )
                
                # Ensure it's a JAX array
                if not isinstance(returns, jnp.ndarray):
                    returns = jnp.array(returns)
            
            else:
                raise TypeError(
                    f"Invalid data format: {type(returns)}. "
                    "Portfolio returns must be pandas DataFrame/Series, numpy array, or JAX array."
                )
            
            # Final validation: ensure output is 1D jnp.Array
            if returns.ndim != 1:
                raise ValueError(
                    f"Portfolio returns must be 1D, got {returns.ndim}D array "
                    f"with shape {returns.shape}. "
                    "This indicates a conversion error."
                )
            
            # Validation successful
            if summary is True: 
                print(f"✓ Portfolio returns validated: shape={returns.shape}, dtype={returns.dtype}")
                print(f"  First 5 values: {returns[:5]}")
            
            # Call the original function with validated returns
            return func(returns, *args, **kwargs)
        
        return wrapper
    
    def simple_validation(returns, summary=False):
            '''The simplier version of the wrapper validate_portfolio_inputs. Works with singular series. '''
            # Check if returns exists
            if returns is None:
                raise ValueError("Portfolio returns data is None. Please provide valid data.")
            
            # Convert pandas DataFrame to jnp.Array
            if isinstance(returns, pd.DataFrame):
                if returns.empty:
                    raise ValueError("Portfolio returns DataFrame is empty. Please check the data source.")
                
                # Check dimensionality before conversion
                if returns.shape[1] > 1:
                    raise ValueError(
                        f"Portfolio returns DataFrame has {returns.shape[1]} columns. "
                        "Expected 1D data. Please compute portfolio returns (e.g., weighted average) "
                        "before passing to this function."
                    )
                
                # Convert single column DataFrame to 1D jnp.Array
                returns = jnp.array(returns.squeeze().values)
            
            # Convert pandas Series to jnp.Array
            elif isinstance(returns, pd.Series):
                returns = jnp.array(returns.values)
            
            # Convert numpy array to jnp.Array
            elif isinstance(returns, np.ndarray):
                if returns.size == 0:
                    raise ValueError("Portfolio returns numpy array is empty.")
                
                # Check dimensionality
                if returns.ndim > 1:
                    raise ValueError(
                        f"Portfolio returns numpy array has {returns.ndim} dimensions "
                        f"with shape {returns.shape}. Expected 1D array. "
                        "Please compute portfolio returns before passing to this function."
                    )
                
                returns = jnp.array(returns)
            
            # Handle JAX arrays (using jax.Array for modern JAX versions)
            elif isinstance(returns, jax.Array):
                if returns.size == 0:
                    raise ValueError("Portfolio returns JAX array is empty.")
                
                # Check dimensionality
                if returns.ndim > 1:
                    raise ValueError(
                        f"Portfolio returns JAX array has {returns.ndim} dimensions "
                        f"with shape {returns.shape}. Expected 1D array. "
                        "Please compute portfolio returns before passing to this function."
                    )
                
                # Ensure it's a JAX array
                if not isinstance(returns, jnp.ndarray):
                    returns = jnp.array(returns)
            
            else:
                raise TypeError(
                    f"Invalid data format: {type(returns)}. "
                    "Portfolio returns must be pandas DataFrame/Series, numpy array, or JAX array."
                )
            
            # Final validation: ensure output is 1D jnp.Array
            if returns.ndim != 1:
                raise ValueError(
                    f"Portfolio returns must be 1D, got {returns.ndim}D array "
                    f"with shape {returns.shape}. "
                    "This indicates a conversion error."
                )
            
            # Validation successful
            if summary is True: 
                print(f"✓ Portfolio returns validated: shape={returns.shape}, dtype={returns.dtype}")
                print(f"  First 5 values: {returns[:5]}")
            
            # Call the original function with validated returns
            return returns
    
    @staticmethod
    def simulated_data_banch(size): 
        _ = []
        for v in [1, 2, 3, 5, 7, 10, 15, 20, 35, 50]: 
            r = stats.t.rvs(v, size=size)
            _.append(r)

        normi = stats.norm.rvs(size=size)
        _.append(normi)
        lap = stats.laplace.rvs(size=size)
        _.append(lap)

        return _ 


@Auxiliary.timer
def portfolio_return(returns, *args):
    '''Returns total return across all the assets in the portfolio (IN WORK)
    
    !!! One must ensure the dimensionality across all returns is the same (# of observatrions) 

    Args: 
        returns: either ND array, where N is the number of assets or an array of returns for a single assets
        *args (optional): arrays of returns of other assets 
         
    Raises: 
        ValueError: if the type of input is not suited for jax.Array conversion '''
    #if the input is a singular NDarray;
    if args is None: 
        
        #if the input is a 1D array return it back
        if returns.shape[0] == 1: 
            return returns
        
        if not isinstance(returns, jax.Array):
            try: 
                returns = jnp.asarray(returns, dtype='float32')
            except: raise ValueError(f"Invalid data format: {type(_)}. "
                    "Portfolio returns must be pandas Series, numpy array, or JAX array.")
        
        else:
            for asset_returns in returns: 
                Auxiliary.simple_validation(asset_returns) 
                _ = jnp.sum(returns, axis=0)
                return _ 
    
    #check that the first argument is of a proper data type 
    try: 
        returns = jnp.array(returns, dtype='float32')
    except: raise ValueError(f"Invalid data format: {type(returns)}. "
            "Portfolio returns must be pandas Series, numpy array, or JAX array.")
    return_list = [returns]
    #if the input consists of several 
    for _ in args: 
            if not isinstance(_, jax.Array):
                try: 
                    asset_return = jnp.array(_, dtype='float32')
                except: raise ValueError(f"Invalid data format: {type(_)}. "
                        "Portfolio returns must be pandas Series, numpy array, or JAX array.")
            elif isinstance(_, jax.Array): 
                asset_return = _

            return_list.append(asset_return)

    portfolio_return = jnp.stack(return_list)
    _ = jnp.sum(portfolio_return, axis=0)

    return _

@jax.jit
def _portfolio_return_fast(returns: jax.Array):
    '''sub-function of portfolio_return_fast()'''
    _ = jnp.sum(returns, axis=0)
    return _ 

@Auxiliary.timer
def portfolio_return_fast(returns: Union[jax.Array, list]):
    '''Uses JAX Just-In-Time calculation for the portfolio return calculation'''
    if isinstance(returns, jax.Array): 
        _portfolio_return_fast(returns)
    else: 
        return_list = []
        for _ in returns: 
            try: 
                asset_return = jnp.array(_, dtype='float32')
            except: raise ValueError(f"Invalid data format: {type(_)}. "
                            "Portfolio returns must be pandas Series, numpy array, or JAX array.")

            return_list.append(asset_return)

        portfolio_return = jnp.stack(return_list)
        _ = _portfolio_return_fast(portfolio_return)
        return _

@dataclass
class GBMParams:
    volatility: Floats 
    mean: Floats 

def make_gbm_simulator(
        params: GBMParams,
        T: int, 
        granularity: int = 1000
    ): 
    '''Creates a Geometric Brownian Motion simulator with predetermined parameters. 
    Returns a batch of simulated data.'''
    def simulate(n_paths: int, seed = None):
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        time_grid = T * granularity
        dt = 1 / granularity
        norm = rng.normal(size=(n_paths, time_grid))

        #Simulate log returns paths using GBM.
        d_log_S = (
            (params.mean - ((params.volatility**2) / 2)) * dt 
            + params.volatility * norm * np.sqrt(dt)
        )
        d_log_S = np.insert(d_log_S, 0, np.zeros(n_paths), axis=1)

        cum_log_returns = np.cumsum(d_log_S, axis=-1)
        cum_returns = np.exp(cum_log_returns) - 1 

        return cum_returns
    return simulate 

def _terminal_returns(simulator: callable, n_paths: int, seed = None): 
    '''Helper method for stripping and returning the last column of the return matrix.'''
    cummulative_returns = simulator(n_paths, seed)
    terminal_returns = cummulative_returns[:, -1] #in case of GBM terminal return will be distributed lognormally
    return terminal_returns


#incorrectly handles volatility - insert variance instead of standard deviation

# class ProtoPortfolio: 
#     def __init__(self, returns, **kwargs): 
#         self.returns = returns #(portfolio) returns
#         try:
#             self.number_of_assets = self.returns.shape[1] 
#         except IndexError: 
#             self.number_of_assets = 1

#         self.kwargs = kwargs.copy()

#     def calibrate(self, horizon=1, **kwargs): 
#         '''Private method implies variance and mean of returns of assets' returns using GARCH(p, q)
        
#         Now it is unable to handle returns across multiple(>1) assets. Also can not handle horizon > 1'''
#         am = arch_model(self.returns, vol='Garch', dist='normal', **kwargs)
#         result = am.fit(disp='off')
#         forecast = result.forecast(horizon=horizon)
#         mu = result.params.get("mu", 0)
#         sigma = forecast.variance.values[0]
#         parameters = GBMParams(volatility=sigma, mean=mu)

#         self.parameters = parameters #for testing 

#         return None
    
#     def _simulate(self, T: int = 10, n_paths: int = 500, granularity: int = 1000):
#         '''Simualtes GBM n_paths times. Outputs simulated array of terminal returns.'''
#         simulator = make_gbm_simulator(self.parameters, T, granularity)
#         simulated_returns = _terminal_returns(simulator, n_paths)

#         return simulated_returns 
    
#     def historical_var(self, alpha=0.01, T: int = 10, **kwargs): 
#         _ = self._simulate(T, **kwargs)
#         return -np.quantile(_, alpha)




@Auxiliary.timer
@Auxiliary.validate_portfolio_inputs
@jax.jit
def historical_var(returns, alpha=0.01):
    """
    Historical VaR at the given alpha level.
    """
    return -jnp.quantile(returns, alpha)

@Auxiliary.timer
@Auxiliary.validate_portfolio_inputs
def historical_es(returns, alpha=0.01): 
    '''Histrorical Expected Shorfall at the given alpha level'''
    _ = -1 * returns 
    sorted_returns = jnp.sort(_) #lossess are to the right
    index = jnp.floor(alpha * sorted_returns.shape[0]).astype(jnp.int32)    
    losses = sorted_returns[index:] 
    es = jnp.mean(losses)
    return es

@Auxiliary.timer
@Auxiliary.validate_portfolio_inputs
def parametric_var_normal(returns, alpha=0.01):
    """
    Parametric VaR assuming normal distribution.
    """
    mean = jnp.mean(returns)
    std = jnp.std(returns)
    var = stats.norm.ppf(alpha, loc=mean, scale=std)
    return -var

def parametric_es_normal(returns, alpha=0.01):
    '''Parametric ES assuming normal distribution
    
    For details consult Hull, J. (2012). Risk management and financial institutions  / John C. Hull. (3rd ed.). John Wiley. Equation (11.2)
    '''
    mean = jnp.mean(returns)
    std = jnp.std(returns)
    y = stats.norm.ppf(alpha)
    es = mean + std*((exp(-(y**2)/2))/(sqrt(2*pi)*(alpha)))
    return es

@Auxiliary.timer
def garch_var(
    returns,
    alpha: float = 0.01,
    p: int = 1,
    q: int = 1,
    mean: str = "Zero",
    vol: str = "Garch",
    dist: str = "normal",
    rescale: bool = False,
    hold_back: int = 0,
    last_obs: Union[int, None] = None,
    update_freq: int = 1,
    starting_values: Union[None, List[float]] = None,
    options: dict = None, 
    disp: str = 'off'
    ):
    """
    Estimates Value at Risk (VaR) using a GARCH model.
    This function fits a GARCH(p, q) model to the provided returns and computes the 
    one-step ahead VaR at the specified significance level (alpha). It supports 
    different mean and volatility models, as well as normal and t-distributions 
    for the innovations.
    Args:
        returns (array-like): Array of returns to fit the GARCH model.
        alpha (float, optional): Significance level for VaR calculation (default is 0.01).
        p (int, optional): Order of the GARCH component (default is 1).
        q (int, optional): Order of the ARCH component (default is 1).
        mean (str, optional): Mean model to use ('Zero', 'Constant', etc.) (default is "Zero").
        vol (str, optional): Volatility model to use ('Garch', etc.) (default is "Garch").
        dist (str, optional): Distribution for innovations ('normal' or 't') (default is "normal").
        rescale (bool, optional): Whether to rescale the returns (default is False).
        hold_back (int, optional): Number of initial observations to exclude from estimation (default is 0).
        last_obs (int or None, optional): Index of the last observation to use in fitting (default is None).
        update_freq (int, optional): Frequency of parameter updates during fitting (default is 1).
        starting_values (list of float or None, optional): Starting values for optimization (default is None).
        options (dict, optional): Options to pass to the optimizer (default is None).
    Returns:
        float: The estimated one-step ahead Value at Risk (VaR) at the specified alpha level (negative value).
    Raises:
        ValueError: If the input parameters are invalid or model fitting fails.
    Example:
        >>> var = garch_var(returns, alpha=0.05, p=1, q=1, dist="t")
    """

    am = arch_model(
        returns,
        mean=mean,
        vol=vol,
        p=p,
        q=q,
        dist=dist,
        rescale=rescale,
        hold_back=hold_back
    )
    res = am.fit(
        last_obs=last_obs,
        update_freq=update_freq,
        starting_values=starting_values,
        options=options,
        disp=disp
    )
    # Forecast 1-step ahead volatility
    forecast = res.forecast(horizon=1)
    mu = res.params.get("mu", 0)
    sigma = forecast.variance.values[-1, 0] ** 0.5
    # Get the quantile for the specified distribution
    if dist == "normal":
        quantile = stats.norm.ppf(alpha)
    elif dist == "t":
        df = res.params.get("nu", 10)
        quantile = stats.t.ppf(alpha, df)
    else:
        quantile = stats.norm.ppf(alpha)  # fallback
    var = mu + sigma * quantile
    return -var

@Auxiliary.timer
@Auxiliary.validate_portfolio_inputs
def garch_es(
    returns,
    alpha: float = 0.01,
    p: int = 1,
    q: int = 1,
    mean: str = "Zero",
    vol: str = "Garch",
    dist: str = "normal",
    rescale: bool = False,
    hold_back: int = 0,
    last_obs: Union[int, None] = None,
    update_freq: int = 1,
    starting_values: Union[None, List[float]] = None,
    options: dict = None, 
    disp: str = 'off'
    ):
    """
    Estimates Expected Shortgall (ES) using a GARCH model.
    This function fits a GARCH(p, q) model to the provided returns and computes the 
    one-step ahead VaR at the specified significance level (alpha). Offspring of garch_var().
    Note, it does not support non-normal distributions. 
    """
    am = arch_model(
        returns,
        mean=mean,
        vol=vol,
        p=p,
        q=q,
        dist=dist,
        rescale=rescale,
        hold_back=hold_back
    )
    res = am.fit(
        last_obs=last_obs,
        update_freq=update_freq,
        starting_values=starting_values,
        options=options,
        disp=disp
    )
    # Forecast 1-step ahead volatility
    forecast = res.forecast(horizon=1)
    mu = res.params.get("mu", 0)
    sigma = forecast.variance.values[-1, 0] ** 0.5
    # Get the quantile for the specified distribution
    if dist == "normal":
        quantile = stats.norm.ppf(alpha)
    else:
        raise ValueError(f'The function does not support {dist} ditribution')
    es = mu + sigma*((exp(-(quantile**2)/2))/(sqrt(2*pi)*(alpha)))
    return es

@Auxiliary.timer
@Auxiliary.validate_portfolio_inputs
def parametric_var_ghd(returns, alpha: float = 0.01):
    """
    Parametric VaR assuming Generalized Hyperbolic Distribution (GHD).
    Uses scipy's genhyperbolic.
    """
    # Fit GHD parameters to the returns
    params = stats.genhyperbolic.fit(returns)
    # Calculate the VaR at the given alpha level
    var = stats.genhyperbolic.ppf(alpha, *params)
    return -var

@jax.jit
@Auxiliary.timer
def GPD_ppf(y, u, beta: float, xi: float): 
    '''Probability densiry function of the Generalized Paretto Distribution (IN WORK).
    
    For details consult Hull, J. (2012). Risk management and financial institutions  / John C. Hull. (3rd ed.). John Wiley. Equation (12.6)

    Args: 
        y (jax.Array): Tail observations coming after (before) the critical value (u)
        u (float): the critical value. Velue of the right (left) tail of the underlying distribution
        beta (float): scale parameter (normalizes cumulitive density function to 1)
        xi (float):  the shape parameter and determines the heaviness of the  tail of the distribution

    Returns: 
        ppf values. jax.Array
    '''
    gamma = y - u 
    denominator = 1/beta 
    power = (-1/xi) - 1
    gamma_scale = xi/beta
    #ppf value 
    g = denominator * ((1 + gamma_scale * gamma ) ** power)

    return g

@Auxiliary.timer
@partial(jax.jit, static_argnames='ppf')
def log_likelyhood(ppf: callable, **kwargs):
    '''Log-likelyhood calculation. ppf parameter allows for arbitrary distribution.'''
    _ = ppf(**kwargs)
    log_densities = jnp.log(_)
    log_l = jnp.sum(log_densities)
    return log_l

@partial(jax.jit, static_argnames='ppf')
def neg_log_likelyhood(ppf: callable, **kwargs):
    '''negative of log_likelyhood'''
    _ = ppf(**kwargs)
    log_densities = jnp.log(_)
    log_l = jnp.sum(log_densities)
    return -log_l

@Auxiliary.timer
def MLE_EVT(returns, u_level = 0.95, mode_var = False): 
    '''Peforms MLE for the EVT (IN WORK)
    
    !!! ADD TYPING 

    Args:
        returns (jax.Array)
        u_level (float): must coincide with the confidence level (1 - alpha) (default: 0.99)
        mode_var (bool): if True returns additional parameters necessary for VaR calculation

    Returns
        results (jax.scipy.optimize.OptimizeResults)
        consult https://docs.jax.dev/en/latest/_autosummary/jax.scipy.optimize.OptimizeResults.html#jax.scipy.optimize.OptimizeResults
    '''
    #choosing the u value
    _ = -1 * returns 
    sorted_returns = jnp.sort(_) #lossess are to the right
    index = jnp.floor(u_level * sorted_returns.shape[0]).astype(jnp.int32)    
    u = sorted_returns[index]
    ys = sorted_returns[index+1:] #the bigger lossess
    n_u = ys.shape[0]

    initial_guess = jnp.array([0.1, jnp.log(0.1)], dtype='float64') #log(beta) was suggested by LLM !!! - it is crucial

    def objective_function(initial_guess, ys, u):
        params = {
            'xi': initial_guess[0],
            'beta': jnp.exp(initial_guess[1]) # guarantees beta > 0
        }
        return neg_log_likelyhood(GPD_ppf, y=ys, u=u, **params)
        
    result = minimize(objective_function, initial_guess, args=(ys, u), method='BFGS')

    if mode_var is True: #for VaR calculation
        return result, u, n_u

    return result 

def EVT_var(returns, alpha=0.01, u_level = 0.95, return_details = False): 
    """
    EVT VaR assuming tail follow GPD (Generalized Paretto Distribution).

    For details refer to Hull, J. (2012). Risk management and financial institutions  / John C. Hull. (3rd ed.). John Wiley. Equation (12.9)
    """
    result, u, n_u = MLE_EVT(returns, u_level=u_level, mode_var=True)

    xi = result.x[0]
    _log_beta = result.x[1]
    beta = jnp.exp(_log_beta)

    nominator = beta/xi 
    power = xi * (-1)
    n = returns.shape[0]

    print('IMPORTANT', n_u, n, beta, xi, result.status)

    VaR = u + nominator*(((n / n_u * alpha)**(power)) - 1)  
    if nominator < 0: 
        print('!!!!!!!!!!FLAG')

    if return_details is True: 
        return VaR, result, u, [beta, xi] #allows for detailed results analysis

    return VaR 

def EVT_es(returns, alpha=0.01, u_level = 0.95, return_details = False): 
    """
    EVT VaR assuming tail follow GPD (Generalized Paretto Distribution).

    For details refer to Hull, J. (2012). Risk management and financial institutions  / John C. Hull. (3rd ed.). John Wiley. Equation (12.10)
    """
    result, u, n_u = MLE_EVT(returns, u_level=u_level, mode_var=True)

    xi = result.x[0]
    _log_beta = result.x[1]
    beta = jnp.exp(_log_beta)

    nominator = beta/xi 
    power = xi * (-1)
    n = returns.shape[0]

    VaR = u + nominator*(((n / n_u * alpha)**(power)) - 1) 

    nominator2 = VaR + beta + (xi * u) 
    denominator = 1 - xi 
    es = nominator2/denominator 

    return es 


class Test: 
    '''Class for .ipynb files and tests'''
    def var_all_methods(returns, alpha, u=0.95): 
        print("Historical VaR:", historical_var(returns, alpha))
        print("Parametric VaR:", parametric_var_normal(returns, alpha))
        print("GARCH(1,1) with Normal distribution VaR:", garch_var(returns, alpha))
        print("GARCH(1,1) with Student distribution VaR:", garch_var(returns, alpha, dist='t'))
        print('EVT VaR', EVT_var(returns, alpha, u_level=u))
        return None 

if __name__ == "__main__":
    key = random.PRNGKey(42)
    # Simulate some returns
    returns = random.normal(key, (5000,)) * 0.01  # 1000 daily returns, mean 0, std 1%
    alpha = 0.01

    print("Historical VaR:", historical_var(returns, alpha))
    print("Parametric VaR:", parametric_var_normal(returns, alpha))
    print("GARCH(1,1) with Normal distribution VaR:", garch_var(returns, alpha))
    print("GARCH(1,1) with Student distribution VaR:", garch_var(returns, alpha, dist='t'))
    # print("Generalized Hyperbolic Distribution VaR:", parametric_var_ghd(returns, alpha))

    returns1 = random.normal(key, (5000,)) * 0.01
    returns2 = random.normal(key, (5000,)) * 0.01
    returns3 = random.normal(key, (5000,)) * 0.01

    portfolio_returns = portfolio_return(returns1, returns2, returns3)
    portfolio_return = portfolio_return_fast([returns1, returns2, returns3])
    returns = jnp.stack([returns1, returns2, returns3])
    portfolio_return = portfolio_return_fast(returns)

    # params = {
    #     'y': returns1, 
    #     'u': 0.02332493543601386,
    #     'beta': 1,
    #     'xi': 1
    # }
    # params2 = {
    #     'beta': 1,
    #     'xi': 1
    # }
    # print(GPD_ppf(**params))
    # print(log_likelyhood(GPD_ppf, **params))
    # print(log_likelyhood(GPD_ppf, y = returns1, u=0.02332493543601386, **params2))

    result = MLE_EVT(returns1)
    print(result.success)
    print(result.x)
    print(result.fun)
    print(result.status)
    print('========================================================================')
    # returns_st = stats.t.rvs(3, size=500000)
    returns_st = stats.t.rvs(5, size=500000)
    print(returns_st)

    _params = {
        'y': returns_st, 
        'u': 0.02332493543601386,
        'beta': 1,
        'xi': 1
    }
    _params2 = {
        'beta': 1,
        'xi': 1
    }
    print(GPD_ppf(**_params))
    print(log_likelyhood(GPD_ppf, **_params))
    print(log_likelyhood(GPD_ppf, y = returns_st, u=0.02332493543601386, **_params2))

    result = MLE_EVT(returns_st)
    print(result.success)
    print(result.x)
    print(result.fun)
    print(result.status)


    print('VaR estimation for model data distrubuted as Student ')
    alpha = 0.01
    print("Historical VaR:", historical_var(returns_st, alpha))
    print("Parametric VaR:", parametric_var_normal(returns_st, alpha))
    print("GARCH(1,1) with Normal distribution VaR:", garch_var(returns_st, alpha))
    print("GARCH(1,1) with Student distribution VaR:", garch_var(returns_st, alpha, dist='t'))
    print('EVT VaR', EVT_var(returns_st, alpha, u_level=0.95))

    print('===============================================================================================================')
    ENABLE_TIMING = False
    alpha = 0.01
    for r in Auxiliary.simulated_data_banch(size=5000): 
        print("Historical VaR:", historical_var(r, alpha))
        print("Parametric VaR:", parametric_var_normal(r, alpha))
        print("GARCH(1,1) with Normal distribution VaR:", garch_var(r, alpha))
        print("GARCH(1,1) with Student distribution VaR:", garch_var(r, alpha, dist='t'))
        print('EVT VaR', EVT_var(r, alpha, u_level=0.95))
        print('-------------------------------------------------------')
    
    print('===============================================================================================================')
    # test = stats.norm.rvs(size=5000)
    test = stats.t.rvs(1, size=5000)
    std = jnp.std(test)
    mean = jnp.mean(test)
    print(mean, std)
    print(historical_var(returns=test, alpha=0.01))
    print(historical_var(returns=test, alpha=0.05))
    VaR, result, u, _ = EVT_var(returns=test, alpha=0.01, u_level=0.95, return_details=True)
    print(VaR)
    print(u) 
    print('-----------------------------------------')
    test = stats.t.rvs(100, size=5000)
    std = jnp.std(test)
    mean = jnp.mean(test)
    print(mean, std)
    print(historical_var(returns=test, alpha=0.01))
    print(historical_var(returns=test, alpha=0.05))
    VaR, result, u, _ = EVT_var(returns=test, alpha=0.01, u_level=0.95, return_details=True)
    print(VaR)
    print(u) 
    print('==========================================================================================')

    std = 20 
    returns_example1110 = stats.norm.rvs(scale=std, size=5000)

    t4 = parametric_es_normal(returns_example1110)
    print(t4)


