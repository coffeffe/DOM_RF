#That file contains some useful methods fore working with GARCH models in the contxt of VaR modeling 

from arch import arch_model
import numpy as np 
from Vares import historical_var, _terminal_returns
from Vares_simulations import SimParams, make_random_path_simulator_local_vol_student, solve_local_vol_gbm_log

def fit_GARCH_VaR(returns, garch_simulation: callable, lag=365, **kwargs): 
    time_series_length = returns.shape[0]
    avialable_lenth = time_series_length - lag
    VaRs = [0] * lag

    for t in range(avialable_lenth): 
        returns_sliced = returns[t:t+365]
        VaRs.append(garch_simulation(returns_sliced))

    return VaRs

#Student GARCH

def student_GARCH_simulation(returns, horizon=10, n_paths=50_000, granularity=1, alpha=0.01): 
    model = arch_model(returns, vol='GARCH', p=1, o=0, q=1, dist='t', rescale=False)
    results = model.fit(disp='off')
    forecast = results.forecast(horizon=horizon)
    mu = results.params.get("mu", 0)
    sigma2 = forecast.variance.values[0]
    sigma = sigma2 ** (1/2) #garch model estimates the variance of the underlyinh sample
    nu = results.params.get('nu', 0)
    parameters = SimParams(volatility=sigma, mean=mu)

    simulator = make_random_path_simulator_local_vol_student(parameters, 10, nu, granularity=granularity)

    sim_returns = _terminal_returns(simulator=simulator, n_paths=n_paths) 
    # print(sim_returns[sim_returns < -0.9])

    # plt.hist(sim_returns, bins=50)
    # plt.show()

    VaR = historical_var(sim_returns, alpha)
    
    return VaR

def normal_GARCH_simulation(returns, horizon=10, n_paths=50_000, granularity=1, alpha=0.01): 
    model = arch_model(returns, vol='GARCH', p=1, o=0, q=1, dist='normal')
    results = model.fit(disp='off')
    forecast = results.forecast(horizon=horizon)
    mu = results.params.get("mu", 0)
    sigma2 = forecast.variance.values[0]
    sigma = sigma2 ** (1/2) #garch model estimates the variance of the underlyinh sample
    # nu = results.params.get('nu', 0)
    parameters = SimParams(volatility=sigma, mean=mu)

    # _ = solve_local_vol_gbm(parameters, 10, n_paths=n_paths)
    sim_returns = solve_local_vol_gbm_log(parameters, 10, n_paths=n_paths)

    # plt.hist(sim_returns, bins=50)
    # plt.show()

    VaR = historical_var(sim_returns, alpha)
    
    return VaR

def EGARCH_simulation(returns, horizon=10, n_paths=50_000, granularity=1, alpha=0.01): 
    model = arch_model(returns, vol='EGARCH', p=1, o=0, q=1, dist='normal')
    results = model.fit(disp='off')
    forecast = results.forecast(horizon=horizon, method='simulation')
    mu = results.params.get("mu", 0)
    sigma2 = forecast.variance.values[0]
    sigma = sigma2 ** (1/2) #garch model estimates the variance of the underlyinh sample
    # nu = results.params.get('nu', 0)
    parameters = SimParams(volatility=sigma, mean=mu)

    # _ = solve_local_vol_gbm(parameters, 10, n_paths=n_paths)
    sim_returns = solve_local_vol_gbm_log(parameters, 10, n_paths=n_paths)

    # plt.hist(sim_returns, bins=50)
    # plt.show()

    VaR = historical_var(sim_returns, alpha)
    
    return VaR

def GJRGARCH_simulation(returns, horizon=10, n_paths=50_000, granularity=1, alpha=0.01): 
    model = arch_model(returns, vol='GARCH', p=1, o=1, q=1, dist='normal')
    results = model.fit(disp='off')
    forecast = results.forecast(horizon=horizon)
    # forecast = results.forecast(horizon=horizon)
    mu = results.params.get("mu", 0)
    sigma2 = forecast.variance.values[0]
    sigma = sigma2 ** (1/2) #garch model estimates the variance of the underlyinh sample
    # nu = results.params.get('nu', 0)
    parameters = SimParams(volatility=sigma, mean=mu)
    print(parameters)

    # _ = solve_local_vol_gbm(parameters, 10, n_paths=n_paths)
    sim_returns = solve_local_vol_gbm_log(parameters, 10, n_paths=n_paths)

    # plt.hist(sim_returns, bins=50)
    # plt.show()

    VaR = historical_var(sim_returns, alpha)
    
    return VaR

def EGARCHo1_simulation(returns, horizon=10, n_paths=50_000, granularity=1, alpha=0.01): 
    model = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='normal')
    results = model.fit(disp='off')
    forecast = results.forecast(horizon=horizon, method='simulation')
    mu = results.params.get("mu", 0)
    sigma2 = forecast.variance.values[0]
    sigma = sigma2 ** (1/2) #garch model estimates the variance of the underlyinh sample
    # nu = results.params.get('nu', 0)
    parameters = SimParams(volatility=sigma, mean=mu)

    # _ = solve_local_vol_gbm(parameters, 10, n_paths=n_paths)
    sim_returns = solve_local_vol_gbm_log(parameters, 10, n_paths=n_paths)

    # plt.hist(sim_returns, bins=50)
    # plt.show()

    VaR = historical_var(sim_returns, alpha)
    
    return VaR


from arch.univariate import GARCH, EWMAVariance, EGARCH, RiskMetrics2006
from arch.univariate import Normal, StudentsT
from arch.univariate import ARX, HARX, ARCHInMean, LS, ConstantMean

__VOLATILITY__ = {'GARCH': GARCH, 'TARCH': GARCH, 'EWMAVariance': EWMAVariance, 'EGARCH': EGARCH}
__DISTRIBUTIONS__ = {'norm': Normal, 't': StudentsT}
__MEAN__ = {'arx': ARX, 'harx': HARX, 'LS': LS, 'a-i-m': ARCHInMean, 'CM': ConstantMean}

def model_construction(mean_process: str, volatiltiy_process: str, dist_process: str, lags, **kwargs): 
    def simulate(returns, horizon=10, n_paths=50_000, granularity=1, alpha=0.01): 
        vol = __VOLATILITY__[volatiltiy_process](**kwargs)
        if mean_process == 'CM' or mean_process == 'LS':
            model = __MEAN__[mean_process](returns)
        if mean_process == 'harx' or mean_process == 'a-i-m' or mean_process == 'arx':
            model = __MEAN__[mean_process](returns, lags=lags)
        model.volatility = vol 
        model.distribution = __DISTRIBUTIONS__[dist_process]()
        results = model.fit(disp='off')
        h = int(horizon)                          
        forecast = results.forecast(horizon=h, method='simulation')          
        # mu = results.params.get("mu", 0) - used in the case of a single mean 
        mu = forecast.mean.values[0]
        sigma2 = forecast.variance.values[0]
        sigma = sigma2 ** (1/2) #garch model estimates the variance of the underlyinh sample
        parameters = SimParams(volatility=sigma, mean=mu)

        if dist_process == 'norm': 
            sim_returns = solve_local_vol_mean_gbm_log(parameters, 10, n_paths=n_paths)
        
        if dist_process == 't': 
            nu = results.params.get('nu', 0)
            simulator = make_random_path_simulator_local_vol_mean_student(parameters, 10, nu, granularity=granularity)
            sim_returns = _terminal_returns(simulator=simulator, n_paths=n_paths) 

        VaR = historical_var(sim_returns, alpha)
        
        return VaR

    return simulate