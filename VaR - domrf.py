#idea is all these methods will be unified in the class in the other .py file

#TODO 
# 1. Validation wrapper 
# 2. Timing wrapper 
# 3. Module fopr converting everything into jnp
# 4. Non-normal VaR parametric models
# 5. Monte Carlo VaR method 
# 6. 

from jax import random
from scipy.stats import *
from typing import Union, List, Literal, TypeAlias
import numpy as np
import pandas as pd
from functools import wraps
from arch import arch_model

import jax.numpy as jnp

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

#might be useful for methods working with the data conversion (in the main class e.g.)
number = Union[int, float]
number_like = Union[List[number], number]
array_like = Union[List[number], np.ndarray]
distributions: TypeAlias = Literal["chauchy", "chi2", "expon", "exponpow", "gamma", "lognorm", "norm", "powerlaw", "rayleigh",
                        "uniform", "t", "gumbel_r", "f"]

def validate_portfolio_inputs(func):
    """
    Decorator to validate and convert portfolio returns to 1D jnp.Array format.
    Handles conversions from pandas DataFrame/Series, numpy arrays, and JAX arrays.
    Raises ValueError if input is not 1D after conversion.
    """ 
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if returns exists
        if self.returns is None:
            raise ValueError("Portfolio returns data is None. Please provide valid data.")

        # Convert pandas DataFrame to jnp.Array
        if isinstance(self.returns, pd.DataFrame):
            if self.returns.empty:
                raise ValueError("Portfolio returns DataFrame is empty. Please check the data source.")
            
            # Check dimensionality before conversion
            if self.returns.shape[1] > 1:
                raise ValueError(
                    f"Portfolio returns DataFrame has {self.returns.shape[1]} columns. "
                    "Expected 1D data. Please compute portfolio returns (e.g., weighted average) "
                    "before passing to this function."
                )
            
            # Convert single column DataFrame to 1D jnp.Array
            self.returns = jnp.array(self.returns.squeeze().values)
        
        # Convert pandas Series to jnp.Array
        elif isinstance(self.returns, pd.Series):
            self.returns = jnp.array(self.returns.values)
        
        # Convert numpy array to jnp.Array
        elif isinstance(self.returns, np.ndarray):
            if self.returns.size == 0:
                raise ValueError("Portfolio returns numpy array is empty.")
            
            # Check dimensionality
            if self.returns.ndim > 1:
                raise ValueError(
                    f"Portfolio returns numpy array has {self.returns.ndim} dimensions "
                    f"with shape {self.returns.shape}. Expected 1D array. "
                    "Please compute portfolio returns before passing to this function."
                )
            
            self.returns = jnp.array(self.returns)
        
        # Handle JAX arrays
        elif isinstance(self.returns, (jnp.ndarray, jnp.DeviceArray)):
            if self.returns.size == 0:
                raise ValueError("Portfolio returns JAX array is empty.")
            
            # Check dimensionality
            if self.returns.ndim > 1:
                raise ValueError(
                    f"Portfolio returns JAX array has {self.returns.ndim} dimensions "
                    f"with shape {self.returns.shape}. Expected 1D array. "
                    "Please compute portfolio returns before passing to this function."
                )
        
        else:
            raise TypeError(
                f"Invalid data format: {type(self.returns)}. "
                "Portfolio returns must be pandas DataFrame/Series, numpy array, or JAX array."
            )
        
        # Final validation: ensure output is 1D jnp.Array
        if self.returns.ndim != 1:
            raise ValueError(
                f"Portfolio returns must be 1D, got {self.returns.ndim}D array "
                f"with shape {self.returns.shape}. "
                "This indicates a conversion error."
            )
        
        # Validation successful
        print(f"✓ Portfolio returns validated: shape={self.returns.shape}, dtype={self.returns.dtype}")
        print(f"  First 5 values: {self.returns[:5]}")
        
        # Call the original function
        return func(self, *args, **kwargs)
    
    return wrapper

def historical_var(returns, alpha=0.01):
    """
    Historical VaR at the given alpha level.
    """
    sorted_returns = jnp.sort(returns)
    index = int(jnp.floor(alpha * len(sorted_returns)))
    return -sorted_returns[index]

def parametric_var_normal(returns, alpha=0.01):
    """
    Parametric VaR assuming normal distribution.
    """
    mean = jnp.mean(returns)
    std = jnp.std(returns)
    var = stats.norm.ppf(alpha, loc=mean, scale=std)
    return -var

# def monte_carlo_var(returns, alpha=0.01, n_sim=10000, key=random.PRNGKey(0)):
#     """
#     Monte Carlo VaR using bootstrapped returns. (In WORK)
#     """
#     idx = random.randint(key, (n_sim,), 0, len(returns))
#     simulated = returns[idx]
#     sorted_sim = jnp.sort(simulated)
#     index = int(jnp.floor(alpha * n_sim))
#     return -sorted_sim[index]
    
def make_bsm_market_simulator(
    ms: MarketState,
    params: BSParams,
    time_stop: float,
    n_steps: int = 200,
):

    def simulate(n_paths: int, seed: int = 0xB0BA_C_3AB0DA):
        dt = (time_stop - ms.time) / n_steps
        random = np.random.default_rng(seed)
        norm = random.normal(size=(n_paths, n_steps))
        d_log_s = (
            (ms.interest_rate - params.volatility ** 2 / 2) * dt
            + params.volatility * norm * np.sqrt(dt)
        )
        d_log_s = np.insert(d_log_s, 0, np.zeros(n_paths), axis=1)
        return ms.stock_price * np.exp(np.cumsum(d_log_s, axis=-1))

    return simulate

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
    options: dict = None
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
    # Convert JAX array to numpy if needed
    if hasattr(returns, "to_py"):
        returns = returns.to_py()
    else:
        returns = jnp.asarray(returns)
        returns = returns.astype(float)
        returns = returns.flatten()
        returns = returns.__array__()

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
        options=options
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

if __name__ == "__main__":
    key = random.PRNGKey(42)
    # Simulate some returns
    returns = random.normal(key, (1000,)) * 0.01  # 1000 daily returns, mean 0, std 1%
    alpha = 0.01

    print("Historical VaR:", historical_var(returns, alpha))
    print("Parametric VaR:", parametric_var_normal(returns, alpha))
    print("Monte Carlo VaR:", monte_carlo_var(returns, alpha, key=key))