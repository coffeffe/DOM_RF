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
import seaborn as sns
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

from Vares import Auxiliary
import Vares

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
    Returns a batch of simulated data. The simulated data is distributed lognormally shifted by -1 term.
    Parameters correspond to ones drown from normal with mean (params.mean - (params.volatility**2) / 2)) and std params.volatility'''
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

#same as make_gbm_simulator_2 but with array slicing is needed here.
@Auxiliary.timer
def make_gbm_simulator_local_vol(
        params: GBMParams, 
        T: int, 
        granularity: int = 1000 
): 
    '''Same as make_gbm_simulator(), but allows for varying daily volatility of the underlying. A more flexible solution'''
    if T < params.volatility.shape[0]: 
        raise ValueError('Quantity of forecasted volatility can not exceed number of days to simulte')
    
    elif T > params.volatility.shape[0]:
        print('Number of simulated days exceed number of forecasted volality points. Simulation will assume constant long-term volatility.')
    
    def simulate(n_paths, seed = None): 
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        n_vol = params.volatility.shape[0]

        dt = 1 / granularity
        time_grid = T * granularity
        norm = rng.normal(size=(n_paths, time_grid))

        #version 2
        # d_log_S = np.zeros((n_paths, ))
        for day in range(0, n_vol): 

            if day == 0: 
                q = (day+1) * granularity
                z_values = norm[:, :q]

                _ = (
                (params.mean - ((params.volatility[day]**2) / 2)) * dt 
                + params.volatility[day] * z_values * np.sqrt(dt)
                )
                d_log_S = np.concatenate((d_log_S, _), axis=1) if day != 0  else _
            if  0 < day < params.volatility.shape[0]:
                q = (day+1) * granularity
                p = day * granularity
                z_values = norm[:, p:q]

                _ = (
                (params.mean - ((params.volatility[day]**2) / 2)) * dt 
                + params.volatility[day] * z_values * np.sqrt(dt)
                )
                d_log_S = np.concatenate((d_log_S, _), axis=1) if day != 0 else _

        p = n_vol * granularity
        _ = (
        (params.mean - ((params.volatility[-1]**2) / 2)) * dt 
        + params.volatility[-1] * norm[:, p:] * np.sqrt(dt)
        )
        d_log_S = np.concatenate((d_log_S, _), axis=1)
        # d_log_S = np.insert(d_log_S, 0, np.zeros(n_paths), axis=1)

        cum_log_returns = np.cumsum(d_log_S, axis=-1)
        cum_returns = np.exp(cum_log_returns) - 1 

        return cum_returns
    return simulate

class ProtoPortfolio: 
    def __init__(self, returns, **kwargs): 
        self.returns = returns #(portfolio) returns
        try:
            self.number_of_assets = self.returns.shape[1] 
        except IndexError: 
            self.number_of_assets = 1

        self.kwargs = kwargs.copy()

    def calibrate(self, horizon=1, **kwargs): 
        '''Private method implies variance and mean of returns of assets' returns using GARCH(p, q)'''
        am = arch_model(self.returns, vol='Garch', dist='normal', **kwargs)
        result = am.fit(disp='off')
        forecast = result.forecast(horizon=horizon)
        mu = result.params.get("mu", 0)
        sigma2 = forecast.variance.values[0]
        sigma = sigma2 ** (1/2) #garch model estimates the variance of the underlyinh sample
        parameters = GBMParams(volatility=sigma, mean=mu)

        self.parameters = parameters #for testing 

        return None
    
    def _simulate(self, T: int = 10, n_paths: int = 5000, granularity: int = 1000):
        '''Simualtes GBM n_paths times. Outputs simulated array of terminal returns.'''
        if self.parameters.volatility.shape[0] == 1: 
            simulator = make_gbm_simulator(self.parameters, T, granularity)
            # simulator = make_gbm_simulator(GBMParams(mean=0.0, volatility=0.2), T, granularity)
        else: 
            simulator = make_gbm_simulator_local_vol(self.parameters, T, granularity)
        simulated_returns = _terminal_returns(simulator, n_paths)

        return simulated_returns 
    
    def historical_var(self, alpha=0.01, T: int = 10, **kwargs): 
        _ = self._simulate(T, **kwargs)
        return -np.quantile(_, alpha)

def solve_simple_gbm(params: GBMParams, 
        T: int,
        n_paths: int = 5000,
        S_0: float = 1):
    '''Alternative to make_gbm_generator. Uses analytical explicit solution for terminal returns output.'''
    if params.volatility.shape[0] != 1: 
        raise ValueError('This method is suited only for constant volatility processes')
    W_T = stats.norm.rvs(scale=T, size=n_paths) #Brownian motion value at the terminal time
    return S_0 * np.exp((params.mean - (params.volatility**2)/2)*T + params.volatility * W_T) - 1

@Auxiliary.timer
def solve_local_vol_gbm(
        params: GBMParams, 
        T: int,
        n_paths: int = 5000,
        S_0: float = 1): 
    '''Same as solve_simple_gbm(), but allows for varying daily volatility of the underlying. A more flexible solution. 
    Works best with regular returns.'''
    
    if T < params.volatility.shape[0]: 
        raise ValueError('Quantity of forecasted volatility can not exceed number of days to simulte')
    elif T > params.volatility.shape[0]:
        print('Number of simulated days exceed number of forecasted volality points. Simulation will assume constant long-term volatility.')

    W_t = stats.norm.rvs(scale=1, size=(n_paths, T))
    d_log_S = np.zeros(shape=(n_paths, T))
    #infering daily price 
    for t in range(params.volatility.shape[0]): 
        d_log_S[:, t:t+1] = ((params.mean - ((params.volatility[t]**2) / 2)) 
                + params.volatility[t] * W_t[:, t:t+1])
    _ = params.volatility.shape[0] 
    d_log_S[:, _:] = ((params.mean - ((params.volatility[-1]**2) / 2)) * (T - _)
                + params.volatility[-1] * W_t[:, _:])
    return S_0 * np.exp(np.sum(d_log_S, axis=1)) - 1

__DISTRIBUTIONS__ = Auxiliary.__DISTRIBUTIONS__

class SimParams(GBMParams): 
    def __init__(self, *args, parameters: GBMParams = None, **kwargs): 
        '''Copies parameters of the already ready GBM parameters'''
        if parameters is None: 
            super().__init__(*args, **kwargs)
        else: 
            self.volatility = parameters.volatility
            self.mean = parameters.mean
        self.other_parameters = kwargs.copy()

def make_random_path_simulator(
        params: SimParams,
        T: int, 
        dist: str = 't',
        granularity: int = 1000,
        **kwargs
    ): 
    '''Simulates returns driven by the prespecified innovation (dist). Kwargs allow for adjusting parameters of particular distributions.
    Returns a batch of simulated data. The simulated data is shifted by -1 term.'''
    def simulate(n_paths: int, seed = None):
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        time_grid = T * granularity
        dt = 1 / granularity
        stoch_comp = __DISTRIBUTIONS__[dist].rvs(**kwargs, size=(n_paths, time_grid))
        # print(stoch_comp)
        if np.inf in stoch_comp: 
            raise MemoryError

        #Simulate log returns paths using GBM.
        d_log_S = (
            params.mean * dt 
            + params.volatility * stoch_comp * np.sqrt(dt)
        )

        cum_log_returns = np.cumsum(d_log_S, axis=-1)
        # return cum_log_returns
        cum_returns = np.exp(cum_log_returns) - 1

        return cum_returns
    return simulate 

def make_random_path_simulator_local_vol(
        params: SimParams,
        T: int, 
        dist: distributions = 'norm', #changed student t to normal distribution ass this method does not handle other distributions well :( 
        granularity: int = 1000,
        **kwargs
    ): 
    '''Simulates returns driven by the prespecified innovation (dist). Kwargs allow for adjusting parameters of particular distributions.
    Returns a batch of simulated data. The simulated data is shifted by -1 term.'''

    if T < params.volatility.shape[0]: 
        raise ValueError('Quantity of forecasted volatility can not exceed number of days to simulte')
    
    elif T > params.volatility.shape[0]:
        print('Number of simulated days exceed number of forecasted volality points. Simulation will assume constant long-term volatility.')

    def simulate(n_paths: int, seed = None):
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        time_grid = T * granularity
        dt = 1 / granularity
        stoch_comp = __DISTRIBUTIONS__[dist].rvs(**kwargs, size=(n_paths, time_grid))

        #iterating through available volatility points 
        d_log_S = np.zeros(shape=(n_paths, time_grid))
        for t in range(params.volatility.shape[0]):
            _t = t * granularity
            d_log_S[:, _t:_t+granularity] = (
                params.mean * dt + 
                params.volatility[t] * stoch_comp[:, _t:_t+granularity] * np.sqrt(t)
            )
        _T = params.volatility.shape[0] * granularity
        d_log_S[:, _T:] = ((params.mean * dt) +
                    + params.volatility[-1] * stoch_comp[:, _T:] * np.sqrt(dt))

        cum_log_returns = np.cumsum(d_log_S, axis=-1)
        # return cum_log_returns
        cum_returns = np.exp(cum_log_returns) - 1

        return cum_returns
    return simulate 

def make_random_path_simulator_local_vol_student(
    params: SimParams,
    T: int, 
    nu: float, 
    dist: distributions = 't',
    granularity: int = 1000,
):
    '''Simulates returns driven by the Student innovation. 
    Returns a batch of simulated data. The simulated data is logged.  Works with the log returns best'''

    if T < params.volatility.shape[0]: 
        raise ValueError('Quantity of forecasted volatility can not exceed number of days to simulte')
    
    elif T > params.volatility.shape[0]:
        print('Number of simulated days exceed number of forecasted volality points. Simulation will assume constant long-term volatility.')

    def simulate(n_paths: int, seed = None):
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        time_grid = T * granularity
        dt = 1 / granularity
        Z = stats.t.rvs(df=nu, size=(n_paths, time_grid)) #the innovation random variabled
        if nu > 2.0: 
            stoch_comp = Z * np.sqrt(nu-2) / np.sqrt(nu) #scaled to have unit variance 
        else: 
            raise ValueError('Can not model the price process as the innovation variable has non-finite variance')

        #iterating through available volatility points 
        d_log_S = np.zeros(shape=(n_paths, time_grid))
        for t in range(params.volatility.shape[0]):
            _t = t * granularity
            d_log_S[:, _t:_t+granularity] = (
                params.mean * dt + 
                params.volatility[t] * stoch_comp[:, _t:_t+granularity] * np.sqrt(t)
            )
        _T = params.volatility.shape[0] * granularity
        d_log_S[:, _T:] = ((params.mean * dt) +
                    + params.volatility[-1] * stoch_comp[:, _T:] * np.sqrt(dt))

        cum_log_returns = np.cumsum(d_log_S, axis=-1)
        return cum_log_returns
 
    return simulate

def make_random_path_simulator_local_vol_log(
        params: SimParams,
        T: int, 
        dist: distributions = 'norm',
        granularity: int = 1000,
        **kwargs
    ): 
    '''Simulates returns driven by the BM innovation (dist). 
    Returns a batch of simulated data. The simulated data is logged. Works with the log returns best.'''

    if T < params.volatility.shape[0]: 
        raise ValueError('Quantity of forecasted volatility can not exceed number of days to simulte')
    
    elif T > params.volatility.shape[0]:
        print('Number of simulated days exceed number of forecasted volality points. Simulation will assume constant long-term volatility.')

    def simulate(n_paths: int, seed = None):
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        time_grid = T * granularity
        dt = 1 / granularity
        stoch_comp = __DISTRIBUTIONS__[dist].rvs(**kwargs, size=(n_paths, time_grid))

        #iterating through available volatility points 
        d_log_S = np.zeros(shape=(n_paths, time_grid))
        for t in range(params.volatility.shape[0]):
            _t = t * granularity
            d_log_S[:, _t:_t+granularity] = (
                params.mean * dt + 
                params.volatility[t] * stoch_comp[:, _t:_t+granularity] * np.sqrt(dt)
            )
        _T = params.volatility.shape[0] * granularity
        d_log_S[:, _T:] = ((params.mean * dt) +
                    params.volatility[-1] * stoch_comp[:, _T:] * np.sqrt(dt))

        cum_log_returns = np.cumsum(d_log_S, axis=-1)
        return cum_log_returns
        # cum_returns = np.exp(cum_log_returns) - 1

        # return cum_returns
    return simulate  
    
@Auxiliary.timer
def solve_local_vol_gbm_log(
        params: GBMParams, 
        T: int,
        n_paths: int = 5000,
        S_0: float = 1,
        seed = None): 
    '''Same as solve_local_vol_gbm(), but handles log returns'''
    
    if T < params.volatility.shape[0]: 
        raise ValueError('Quantity of forecasted volatility can not exceed number of days to simulte')
    elif T > params.volatility.shape[0]:
        print('Number of simulated days exceed number of forecasted volality points. Simulation will assume constant long-term volatility.')

    W_t = stats.norm.rvs(scale=1, size=(n_paths, T), random_state=seed)
    d_log_S = np.zeros(shape=(n_paths, T))
    #infering daily price 
    for t in range(params.volatility.shape[0]): 
        d_log_S[:, t:t+1] = (params.mean 
                + params.volatility[t] * W_t[:, t:t+1])
    _ = params.volatility.shape[0] 
    d_log_S[:, _:] = (params.mean * (T - _)
                + params.volatility[-1] * W_t[:, _:])
    
    cum_log_returns = np.sum(d_log_S, axis=-1)
    return cum_log_returns
