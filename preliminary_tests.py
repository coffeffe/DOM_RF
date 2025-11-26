from Vares_simulations import (make_random_path_simulator_local_vol_mean_student, 
                               solve_local_vol_mean_gbm_log, 
                               SimParams, 
                               _terminal_returns,
                               make_random_path_simulator_local_vol_mean_generalized_normal)

from arch.univariate import GARCH, EWMAVariance, EGARCH, RiskMetrics2006, FIGARCH, APARCH, HARCH
from arch.univariate import Normal, StudentsT, GeneralizedError
from arch.univariate import ARX, HARX, ARCHInMean, LS, ConstantMean

from Vares import historical_var

__VOLATILITY__ = {'GARCH': GARCH, 'TARCH': GARCH, 'EWMAVariance': EWMAVariance, 'EGARCH': EGARCH, 'FIGARCH': FIGARCH, 
                  'APARCH': APARCH, 'HARCH': HARCH, 'RM2006': RiskMetrics2006}
__DISTRIBUTIONS__ = {'N': Normal, 'T': StudentsT, 'GE': GeneralizedError}
__MEAN__ = {'AR': ARX, 'HAR': HARX, 'LS': LS, 'a-i-m': ARCHInMean, 'CM': ConstantMean}

def model_construction(mean_process: str, volatiltiy_process: str, dist_process: str, lags=None, forecasting_mode='default', **kwargs): 

    #checks
    if mean_process not in __MEAN__.keys():
        raise TypeError('Unavailable mean process format')
    
    if volatiltiy_process not in __VOLATILITY__.keys():
        raise TypeError('Unavailable volatility process format')
    
    if dist_process not in __DISTRIBUTIONS__.keys():
        raise TypeError('Unavailable distribution name')
    
    if lags is None and mean_process in ['HAR', 'AR']: 
        raise TypeError('Lags are needed for mean processes')
    
    if forecasting_mode not in ['default', '1D']:
        raise TypeError('Unavaiable forecasting mode')

    def simulate(returns, 
                 T=10, 
                 n_paths=50_000, 
                 granularity=1, 
                 alpha=0.01): 
        '''Simulates returns. Allows for dynamic mean. The simulated data MUST logged.'''

        #model specification
        if volatiltiy_process == 'HARCH': #harch inputs are colliding with the mean model inputs :(
            vol = __VOLATILITY__[volatiltiy_process](lags)
        else: 
            vol = __VOLATILITY__[volatiltiy_process](**kwargs)

        if mean_process == 'CM' or mean_process == 'LS':
            model = __MEAN__[mean_process](returns)
        if mean_process == 'HAR' or mean_process == 'AR':
            model = __MEAN__[mean_process](returns, lags=lags)

        model.volatility = vol 
        model.distribution = __DISTRIBUTIONS__[dist_process]()

        #parameter estimation
        results = model.fit(disp='off')

        if forecasting_mode == 'dafault':
            h = int(T) 
        else: h=1           

        #simulation forecast method is waterproof
        forecast = results.forecast(horizon=h, method='simulation') #consider adding analytical forecast     
        mu = forecast.mean.values[0]
        sigma2 = forecast.variance.values[0]
        sigma = sigma2 ** (1/2) #garch model estimates the variance of the underlyinh sample
        parameters = SimParams(volatility=sigma, mean=mu)

        #simulations
        if dist_process == 'N': 
            sim_returns = solve_local_vol_mean_gbm_log(parameters, T, n_paths=n_paths)
        
        if dist_process == 'T': 
            nu = results.params.get('nu', 0)
            simulator = make_random_path_simulator_local_vol_mean_student(parameters, T, nu, granularity=granularity)
            sim_returns = _terminal_returns(simulator=simulator, n_paths=n_paths) 

        if dist_process == 'GE':
            nu = results.params.get('nu', 0)
            simulator = make_random_path_simulator_local_vol_mean_generalized_normal(parameters, T, nu, granularity=granularity)
            sim_returns = _terminal_returns(simulator=simulator, n_paths=n_paths) 

        VaR = historical_var(sim_returns, alpha)
        
        return VaR

    return simulate

def get_volatility_naming():
    return __VOLATILITY__

def get_distributions_naming():
    return __DISTRIBUTIONS__

def get_mean_naming():
    return __MEAN__

__MODELS__ = ['GARCH-CM', 'EGARCH-CM', 'HARCH-CM', 'RM2006-CM', 'EWMAVariance-CM', 
              'APARCH-CM', 'TARCH-CM', 'FIGARCH-CM', 'GARCH-HAR', 'GARCH-AR']

garch_data = [[1,0,1], [2,0,2], [5,0,5], [1,1,1], [2,1,2]]
garch_parameters = [dict(zip(['p', 'o', 'q'], triple)) for triple in garch_data]

aparch_data = [[1,0,1], [2,0,2], [5,1,5]]
aparch_parameters = [dict(zip(['p', 'o', 'q'], triple)) for triple in aparch_data]

figarch_data = [[1, 2.0, 1], [1, 1.0, 1], [1, 1.5, 1], [1, 1.0, 1]]
figarch_parameters = [dict(zip(['p', 'power', 'q'], triple)) for triple in figarch_data]


__MODELPARAMETERS__ = {'GARCH-CM': garch_parameters,
                       'EGARCH-CM': [{'p': 1, 'q': 1}], 
                       'HARCH-CM': [{'lags': [1, 5, 22]}], 
                       'RM2006-CM': [{'tau0': [1560], 'tau1': [4], 'kmax': [14], 'rho': '1.4142135623730951'}],
                       'EWMAVariance-CM': [{'lam': None}], 
                       'APARCH-CM': aparch_parameters,
                       'TARCH-CM': [{'p': 1, 'o': 0, 'q': 1, 'power': 1.0}, {'p': 1, 'o': 1, 'q': 1, 'power': 1.0}],
                       'FIGARCH-CM': figarch_parameters,
                       'GARCH-HAR': [{'p': 1, 'q': 1}], 
                       'GARCH-AR': [{'p': 1, 'q': 1}]
                       }

__MEANPARAMETERS__ = [{'lags': lags} for lags in [1, 5, [1, 5, 22]]]

def get_modelparameters():
    return __MODELPARAMETERS__

def get_meanparameters(): 
    return __MEANPARAMETERS__

def fit_GARCH_VaR(returns, garch_simulation: callable, lag=365, **kwargs): 
    time_series_length = returns.shape[0]
    avialable_lenth = time_series_length - lag
    VaRs = [0] * lag

    for t in range(avialable_lenth): 
        returns_sliced = returns[t:t+365]
        VaRs.append(garch_simulation(returns_sliced))

    return VaRs

import pickle

def modeling(returns, distribution: str): 
    pickle_names = []

    if distribution  not in __DISTRIBUTIONS__: 
        raise TypeError('Unavailable distribution')
    
    for model_type in __MODELS__: #iterate over all model types 
        volatility, mean = model_type.split('-')

        if volatility == 'HARCH' and mean in ['AR', 'HAR']: #colision of the input names
            continue #skip that pair 
        
        for parameters in __MODELPARAMETERS__[model_type]: 
            
            if mean == 'CM': #constant mean does not require lag parameters 
                simulator = model_construction(mean, volatility, distribution, lags=None, **parameters)
                _vars = fit_GARCH_VaR(returns, simulator)

                #creating pickle file with transparent naming
                pickle_name = f'{volatility}-{mean}-{distribution}-{parameters}'
                with open(pickle_name, "wb") as f:
                    pickle.dump(_vars, f)

                pickle_names.append(pickle_name)
            
            else: 
                for mean_parameters in __MEANPARAMETERS__: #iteration over lags of the  dynamic mean models
                    simulator = model_construction(mean, volatility, distribution, lags=mean_parameters, **parameters)
                    _vars = fit_GARCH_VaR(returns, simulator)

                    #creating pickle file with transparent naming
                    pickle_name = f'{volatility}-{mean}({mean_parameters})-{distribution}-{parameters}'
                    with open(pickle_name, "wb") as f:
                        pickle.dump(_vars, f)

                    pickle_names.append(pickle_name)

    #storing pickled names seperately for easy access
    with open('PICKLE_NAMES.pkl', 'wb') as f: 
        pickle.dump(pickle_names)

    return pickle_names
