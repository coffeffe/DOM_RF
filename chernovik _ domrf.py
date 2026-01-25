import os 
import numpy as np  
import numpy.typing as npt
from typing import Union
from datetime import date, datetime, timedelta
import time 
import functools
from dataclasses import dataclass, fields
import inspect
from dateutil.relativedelta import relativedelta
import jax.numpy as jnp
import jax 

@dataclass
class Quotes(): 
    '''
    Stores the current swap market quotes. (IN WORK)

    !!!Assess the necessity and design addition of other contract parameters such as trading TotalBidQty, TotalAskQty etc.  

    Parameters:
        quotes_dtype (np.dtype): custom numpy datatype for storing quotes. Consisits of date of expiration (datetime.date),
            bid (float), ask (float) and mid (float)
        irs_keyrate (quotes_dtype): stores the quotes for the IRS Keyrate contract 
        ois_rounia (quotes_dtype): stores the quotes for the OIS RUONIA contact
        ois_rusfarcny (quotes_dtype): stores the quotes for the OIS RUSFARCNY contract
        xccy (quotes_dtype): stores the quotes for the XCCY RUONIA CNYRUB contract
    
    Public methods:
        update(): updates market quotes in accordance with the current date. (IN WORK)
        manual_update(): allows for a manual update of market quotes. 
    '''
    quotes_dtype = np.dtype([
        ('MaturityDate', 'datetime64[D]'), 
        ('Bid', 'f4'),
        ('Ask', 'f4'), 
        ('Mid', 'f4')
    ])
    irs_keyrate = np.array(0, dtype=quotes_dtype)
    ois_ruonia = np.array(0, dtype=quotes_dtype)
    ois_rusfarcny = np.array(0, dtype=quotes_dtype)
    xccy = np.array(0, dtype=quotes_dtype)
    
    def update(self): 
        '''
        Updates market quotes in accordance with the current state. (IN WORK)

        !!!Requires addition of parsing values for quotes.
        '''
        pass

    def manual_update(self, curve_name: str): 
        '''
        Allows for a manual update of market quotes. 

        Args: 
            curve_name (str): the name of the curve to be updated. Should with the corresponing class variable
        '''

    @staticmethod
    def termination_dates(start_date): 
        '''(IN WORK) currently contains mistakes. Maybe insert own function as timdeltas or something. '''
        if start_date is None: 
            start_date = np.datetime64('today', 'D')

        start_date = np.datetime64(start_date, 'D')

        termination_m = ['1W', '2W', '1M', '2M', '3M', '6M', '9M', '1Y', 
                               '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y']
        
        terminations_dates_ = [0]*len(termination_m)

        terminations_dates_[0] = start_date + np.timedelta64(1, 'W')
        terminations_dates_[1] = start_date + np.timedelta64(2, 'W')

        terminations_dates_[2] = start_date + np.timedelta64(1, 'M')
        terminations_dates_[3] = start_date + np.timedelta64(2, 'M')
        terminations_dates_[4] = start_date + np.timedelta64(3, 'M')
        terminations_dates_[5] = start_date + np.timedelta64(6, 'M')
        terminations_dates_[6] = start_date + np.timedelta64(9, 'M')

        for i in range(7, len(terminations_dates_) + 1): 
            terminations_dates_[i] = start_date + np.timedelta64(i-6, 'Y')

        termination_dates = dict(zip(termination_m, terminations_dates_))
        return terminations_dates_
    

class Quote(): 
    '''
     Stores the current swap market quote for a chosen instrument. Requires complete rework. (IN WORK)

     !!!Assess the necessity and design addition of other contract parameters such as trading TotalBidQty, TotalAskQty etc.  

     Parameters:
        quotes_dtype (np.dtype): custom numpy datatype for storing quotes. Consisits of date of expiration (datetime.date),
            bid (float), ask (float) and mid (float)
        quote (np.array): stores the quote for a chosen instrument 
        pp (bool): 1 if the quoted value are the percentage points, 0 else
    
     Public methods:
        update(): updates market quotes in accordance with the current date. (IN WORK)
     '''
    
    quotes_dtype = jnp.dtype([
        ('MaturityDate', 'datetime64[D]'), 
        ('Bid', 'f4'),
        ('Ask', 'f4'), 
        ('Mid', 'f4')
    ])
    quotes_dtype_test = jnp.dtype([
        ('MaturityDate', 'datetime64[D]'), 
        ('Ask', 'f4')
    ])

    def __init__(self, quote, pp:bool): 
        self.quote = jnp.array(quote, dtype=Quote.quotes_dtype_test)
        self.pp = pp 

    def __new__(cls, quote, *args, **kwargs):
        try: 
            quote = jnp.array(quote, dtype=Quote.quotes_dtype_test)
        except: 
            raise ValueError("Unsupported datatype for class Quote. Reinitialize the instance. ")
        
    def update(self):
        '''
        Updates market quotes in accordance with the current state. (IN WORK)

        !!!Requires addition of parsing values for quotes.
        '''

        #self.quote = 
        #  
        print('UPDATE FUNCTION REQUIRES WORK')
        pass

class Testing:
    '''
    Class storing variables and methods used in the process of development of multicurve pricing script. 
    Its use will be redundant upon deployment 

    Parameters: 
        termination_dates_dict (dictionary): stores maturity dates corresponding to tenors. The reference  effective start date is 19.09.2025
        ois_ruonia_mid (list): stores OIS RUONIA Mid quotes quoted at 18.09.2025
        irs_keyrate_mid (list): stores IRS KEYRATE Mid quotes quoted at 18.09.2025

    Public methods: 
        timer(): Decorator function for timing function calling.
    '''
    termination_dates_dict = {
        '1W': '2025-09-26',
        '2W': '2025-10-03',
        '1M': '2025-10-20',
        '2M': '2025-11-19',
        '3M': '2025-12-19',
        '6M': '2026-03-19',
        '9M': '2026-06-19',
        '1Y': '2026-09-21',
        '2Y': '2027-09-20',
        '3Y': '2028-09-19',
        '4Y': '2029-09-19',
        '5Y': '2030-09-19',
        '6Y': '2031-09-19',
        '7Y': '2032-09-20',
        '8Y': '2033-09-19',
        '9Y': '2034-09-19',
        '10Y': '2035-09-19'
    }

    ois_ruonia_mid = [0.169807, 0.1695, 0.169369, 0.167535, 0.1663, 0.161693, 0.154318, 
                      0.148595, 0.138732, 0.137346, 0.137007, 0.136437, 0.136763, 0.137155, 
                      0.137194, 0.137225, 0.13757]
    
    irs_keyrate_mid = [0.166034, 0.160584, 0.153732, 0.14843, 0.138068, 0.137012, 
                       0.136533, 0.136273, 0.136042, 0.13682, 0.136882, 0.137038, 0.1372459]

    @staticmethod
    def timer(func): 
        ''' Decorator function for timing function calling. '''
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            value = func(*args, **kwargs)
            total = time.perf_counter() -start
            print(f'{func.__name__}() took {total:.10f}s')
            return value 
        return wrapper

if __name__ == '__main__': 
    a = [0, 1, 2, 3, 4] 
    b = a[:2]
    a[:2] = [8]*len(a[:2]) 
    print(a)

    @Testing.timer
    def way_1(array, end, insertion=8): 
        start = 0
        array_copy = array

        array_copy[start:end] = [insertion]*len(array_copy[start:end])
        return array_copy
    
    @Testing.timer
    def way_2(array, end, insertion=8): 
        start = 0
        array_copy = array

        for i in range(end): 
            array_copy[i] = insertion
        return array_copy



    test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    test2 = [1, 2, 3, 4]
    var1, var2, var3, var4 = test2 
    print(var1, var2, var3, var4)

    print((date(2025, 1, 10) - date(2025, 1, 1)).days)\
    
    a = jnp.array()
    print(type(a))