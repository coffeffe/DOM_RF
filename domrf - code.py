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

#to see unfinished code type in "IN WORK"

'''
List of problems: 
1. Initialization and parsing of quotes
2. Do termination dates for each maturity coincide across every type of contract traded (СПФИ)? 
    2.1. If yes: create separate variable for the collection of tenors and termination dates
3. class Quote requires rework. 
4. Numpy and datetime 
5. Quote logic 
'''

'''
To-do list: 
1. Termination date function
2. Dates functions and numpying 
3. Prepare all the calendars
'''

class Solver: 
    '''(IN WORK)'''
    pass

@dataclass
class Curves: 
    '''(IN WORK)'''
    pivot_points: Union[list[date], jax.Arra    y]
    values: Union[list, jax.Array]   
    pass 

@dataclass
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
            print(f'{func.__name__}() took {total:.6f}s')
            return value 
        return wrapper

@dataclass
class Market: 
    '''
    Stores current market parameters used at a wide range of calculations.

    Public methods: 
        update(): updates market paramaters in accordance with the current state. (IN WORK)

    Parameters: 
         today (datetime.date): current date 
         key_rate_0 (float): current CB key rate 
         rub_cny_spot (float): rub/cny spot rate at the current date
    '''
    today_: date
    key_rate_0: float
    rub_cny_spot: float
    termination_dates: jax.Array

    def update(self): 
        '''
        Updates market paramaters in accordance with the current state. (IN WORK)

        !!!Requires addition of parsing values for rub/sny spot rate and key rate. 
        '''

        self.today_ = date.today()
        #self.ley_rate_0 = 
        #self.rub_cny_spot = 
        #figure out a way to parse the values from a source trusted by domrf firewall

class Dates: 
    '''
    Operations with dates (IN WORK)

    Parameters: 
        today_ (datetime.date): stores the starting date for the continuum
        span (int): # of months to be stored 

    Public methods: 
        space(): outputs the continuum of the dates starting today and lasting for 121 months. REDUNDANT 
        custom_space(): outputs the continuum of dates starting with the specifyed date and lasting for specifyed number of months. REDUNDANT
        custom_space_np(): same as custom_space(), but return np.NDArray 
        frac_year(): calculates the year fraction based on the convension of choise
        business_day_np(): converts the date (or a series of dates) in accordance with the business day convention of choice (numpy format)
        business_day(): converts the date (or a series of dates) in accordance with the business day convention of choice (datetime.date)
        calendar(): generates the calendar spanning next 10 years starting today (space_np)
        is_leap_year(): checks wether a year is a leap one 
        termination_dates_dict(): Return the termination dates dictionary for the predetermined start date

    Private methods: 
        _daterange(): generator for the datetime.date format
    '''

    DatetimeArray = npt.NDArray[np.datetime64]
    Datetimes = Union[np.datetime64, DatetimeArray]


    def __init__(self, today_, span_months): 
        self.today_ = today_
        self.span_months = span_months

    @staticmethod
    @Testing.timer
    def space(span_months: int = 121): 
        '''
        Outputs the continuum of dates starting with today and lasting for 121 months. Redundant.

        Args: 
            span_months: the length of the continuum in months  
        
        Returns: 
            dates (list): list of datetime.date variables spanning across ~10 years starting today
        '''
        today_ = date.today()
        span_days = span_months * 31
        end_date = today_ + timedelta(days=span_days)
        dates = [_ for _ in Dates._daterange(today_, end_date)]
        return dates
    
    @staticmethod
    def calendar():
        '''Generates the calendar spanning next 10 years starting today (space_np)'''
        today_ = date.today()
        span_days = 121 * 31
        end_date = today_ + timedelta(days=span_days)
        iterable = [_ for _ in Dates._daterange(today_, end_date)]
        dates = np.fromiter(iterable, dtype='datetime64[D]')
        return dates

    @staticmethod
    @Testing.timer
    def custom_space(start_date: date, span: int): 
        '''
        Outputs the continuum of dates starting with the specifyed date and lasting for specifyed number of months. See space(). Redundant 
        '''
        span_days = span*31
        end_date = start_date + timedelta(days=span_days)
        dates = [_ for _ in Dates._daterange(start_date, end_date)]
        return dates
    
    @staticmethod
    def custom_space_np(start_date: date, span: int):
        '''See custom_space() Returns numpy array. '''
        span_days = span*31
        end_date = start_date + timedelta(days=span_days)
        iterable = [_ for _ in Dates._daterange(start_date, end_date)]
        dates = np.fromiter(iterable, dtype='datetime64[D]')
        return dates

    @staticmethod
    def frac_year(start_date: date, end_date: date, convension: int):
        '''
        Calculates the year fraction based on the convension of choice. Resembles excel YEARFRAC().
        The intervals is [;). I.e. from 2025.01.01 to 2026.01.01 ACT/ACT year_frac will be equal to 1.0

        Args: 
            start_date
            end_date 
            convenision: 0 - 30/360 (American) w/o EOM rule 
                         1 - ACT/ACT (ISDA)
                         2 - ACT/360
                         3 - ACT/365
                         4 - 30E/360 (European)

        For 30/360 the following formula is utilized \frac{360(Y_2-Y-1) + 30(M_2-M_1) + (D_'2 - D_'1)}{360} 
        For details on that consult https://en.wikipedia.org/wiki/Day_count_convention section 30/360

        Returns: 
            Year fraction (in accordance with convension)

        Raises: 
            ValueError: if convension output is not 0, 1, 2, 3, 4
        '''
        if convension not in [0, 1, 2, 3, 4]: 
            raise ValueError('Inappropriate value for convension. \n 0 - 30/360 (American) ' 
                '\n 1 - ACT/ACT \n 2 - ACT/360 \n 3 - ACT/365 \n 4 - 30/360 (European)')
        
        year_fraction = 0 #return
        
        if convension in [0, 4]:
            d1 = start_date.day 
            d2 = end_date.date

            if convension == 0: 
                d1_ = 30 if d1 == 31 else d1 
                d2_ = 30 if d1 == 30 or d1 == 31 else d2 

            if convension == 4: 
                d1_, d2_ = min(d1, 30), min(d2, 30)

            y1 = start_date.year
            y2 = end_date.year
            m1 = start_date.months
            m2 = end_date.months 

            year_fraction = (360*(y2-y1) + 30*(m2 - m1) + (d2_ - d1_))/360
            return year_fraction
        
        #Code proceeds for the ACT/X type convensions 
        delta_ =  end_date - start_date
        delta = delta_.days

        if convension == 2: 
            year_fraction = delta/360 
            return year_fraction
        
        if convension == 3: 
            year_fraction = delta/365
            return year_fraction

        if convension == 1: 
            y1 = start_date.year
            y2 = end_date.year

            year_delta = max(0, y2 - y1 - 1) 

            if year_delta == 0: 
                
                if y2 == y1: 
                    year_fraction = delta/366 if Dates.is_leap_year(y1) else delta/365
                    return year_fraction
                
                if y2 > y1: 
                    #both y1 and y2 can not be leap years simlutaniously 
                    dc1 = (date(y1, 12, 31) + timedelta(days=1) - start_date).days #first year day count 
                    dc2 = (end_date - date(y2, 1, 1)).days #last year day count

                    if Dates.is_leap_year(y2) is True: 
                        year_fraction = (dc1/365) + (dc2/366)
                        return year_fraction

                    if Dates.is_leap_year(y1) is True: 
                        year_fraction = (dc1/366) + (dc2/365)
                        return year_fraction
                    
                    year_fraction = (dc1/365) + (dc2/365)
                    return year_fraction
                
            if year_delta != 0: 
                year_fraction = 0
                year_fraction += year_delta 
                dc1 = (date(y1, 12, 31) + timedelta(days=1) - start_date).days #first year day count 
                dc2 = (end_date - date(y2, 1, 1)).days #last year day count

                if Dates.is_leap_year(y1) is True:
                    year_fraction += dc1/366
                else: year_fraction += dc1/365

                if Dates.is_leap_year(y2) is True: 
                    year_fraction += dc2/366 
                else: year_fraction += dc2/365 

                return year_fraction 

    @staticmethod
    def is_leap_year(year: int): 
        '''Determines whether the year is leap or not.'''
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0): 
            return True 
        else: False 

    @staticmethod
    def _daterange(start_date: date, end_date: date, step_days: int = 1, insclusive=True):
        '''
        Private method used generator for the datetime.date format

        Args: 
            start: initial value for the generating process 
            end : last value of the list 
            step_days: parameter governing the step of the generator process. Represents the number of days b/w each value. 
            inclusive (bool): True if the end date is included, else it is not

        Yields: 
            current (datetime.date): stream of datetime.date variables in the interval [start;end] or [start;end)
        '''
        step = timedelta(days=step_days)
        current = start_date
        last = end_date if insclusive else (end_date - step)
        while current <= last: 
            yield current 
            current += step 

    @staticmethod
    def business_day_np(dates: Datetimes, convention: Union[str, int]): 
        '''Converts the date (or a series of dates) in accordance with the business day convention of choice
        

        !!! Implement the check for the main holidays
        
        Args: 
            dates (Datetimes): dates to be converted
            cconventiononvension: 0 or "FOLLOWING" - following convention ensures the date is moved to the closest working day
                        1 or "MODFOLLOWING" -  following convention ensures the date is moved to the closest working day within a month

        Returns: 
            Dates in accordance with conventions
        '''

        if convention == 0 or convention == "FOLLOWING": 
            dates = np.asarray(dates, dtype='datetime64[D]').copy()

            mask1_0 = np.is_busday(dates, weekmask='Sat')
            dates[mask1_0] += np.timedelta64(2, 'D')
            mask2_0=  np.is_busday(dates, weekmask='Sun')
            dates[mask2_0] += np.timedelta64(1, 'D')

            return dates 
        
        if convention == 1 or convention == "MODFOLLOWING": 
            dates = np.asarray(dates, dtype='datetime64[D]').copy()

            mask1_1 = np.is_busday(dates, weekmask='Sat') 
            mask2_1 = np.is_busday(dates, weekmask='Sun')

            M1 = dates + np.timedelta64(2, 'D')
            M2 = dates + np.timedelta64(1, 'D')

            month = dates.astype('datetime64[M]')
            mask_M1_1 = M1.astype('datetime64[M]') == month
            mask_M2_1 = M2.astype('datetime64[M]') == month

            dates[mask1_1 &  mask_M1_1] += np.timedelta64(2, 'D')  
            dates[mask1_1 & ~mask_M1_1] -= np.timedelta64(1, 'D') 

            dates[mask2_1 &  mask_M2_1] += np.timedelta64(1, 'D') 
            dates[mask2_1 & ~mask_M2_1] -= np.timedelta64(2, 'D')  

            return dates
        
    @staticmethod
    def business_day(dates: list[date], convention: Union[str, int]): 
        '''Converts the date (or a series of dates) in accordance with the business day convention of choice. Twin of business_day().
    
    !!! Implement the check for the main holidays
    
    Args: 
        dates (List[date]): list of datetime.date objects to be converted
        convention: 0 or "FOLLOWING" - following convention ensures the date is moved to the closest working day
                    1 or "MODFOLLOWING" - modified following convention ensures the date is moved to the closest 
                                          working day within a month
    
    Returns: 
        List[date]: Dates in accordance with conventions
    '''
        
        # Ensure dates is a list and make a copy to avoid mutating the original
        dates_copy = [d for d in dates] if isinstance(dates, list) else [dates]
        
        if convention == 0 or convention == "FOLLOWING":
            result = []
            for d in dates_copy:
                # Check if date falls on Saturday (weekday() == 5)
                if d.weekday() == 5:  # Saturday
                    result.append(d + timedelta(days=2))
                # Check if date falls on Sunday (weekday() == 6)
                elif d.weekday() == 6:  # Sunday
                    result.append(d + timedelta(days=1))
                else:
                    result.append(d)
            return result
        
        elif convention == 1 or convention == "MODFOLLOWING":
            result = []
            for d in dates_copy:
                adjusted_date = d
                
                # Check if date falls on Saturday
                if d.weekday() == 5:  # Saturday
                    forward_date = d + timedelta(days=2)  # Move to Monday
                    backward_date = d - timedelta(days=1)  # Move to Friday
                    
                    # Check if forward date is in the same month
                    if forward_date.month == d.month:
                        adjusted_date = forward_date
                    else:
                        adjusted_date = backward_date
                
                # Check if date falls on Sunday
                elif d.weekday() == 6:  # Sunday
                    forward_date = d + timedelta(days=1)  # Move to Monday
                    backward_date = d - timedelta(days=2)  # Move to Friday
                    
                    if forward_date.month == d.month:
                        adjusted_date = forward_date
                    else:
                        adjusted_date = backward_date
                
                result.append(adjusted_date)
            return result
        
        else:
            raise ValueError(f"Invalid convention: {convention}. Use 0/'FOLLOWING' or 1/'MODFOLLOWING'")

    @staticmethod
    def terminations_dates_dict(start_date: date):
        '''Returns dictionary of the terminaton dates for the standard maturities'''
        termination_m = ['1W', '2W', '1M', '2M', '3M', '6M', '9M', '1Y', 
                        '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y']

        terminations_dates_ = [0]*len(termination_m)

        terminations_dates_[0] = start_date + relativedelta(weeks=1)
        terminations_dates_[1] = start_date + relativedelta(weeks=2)

        terminations_dates_[2] = start_date + relativedelta(months=1)
        terminations_dates_[3] = start_date + relativedelta(months=2)
        terminations_dates_[4] = start_date + relativedelta(months=3)
        terminations_dates_[5] = start_date + relativedelta(months=6)
        terminations_dates_[6] = start_date + relativedelta(months=9)

        for i in range(7, len(terminations_dates_)):
            terminations_dates_[i] = start_date + relativedelta(years=(i-6))

        termination_dates_dict = dict(zip(termination_m, terminations_dates_))
        return termination_dates_dict

class Calendar: 
    '''Class is designed to store calendar. Contains calendar for following and modfollowing dates. (IN WORK)'''
    today_ = datetime.today()



    def __init__(self, start_date):
        if start_date is None: 
            start_date = Calendar.today_
        pass

    @staticmethod
    def gorinich(calendar): 
        calendar_np = np.array(calendar, dtype='datetime64[D]')
        _ = jnp.array(dates_np, dtype='int')
        calendar_int = _ - (_[0] - 1) 
        return calendar, calendar_np, calendar_int

class Quote: 
    pass

@Testing.timer
def curve(pivot_points: Union[list[date], jax.Array], values: Union[list, jax.Array], calendar): 
    '''(IN WORK)'''
    if len(pivot_points) != len(values): 
        raise ValueError('There should be as much pivot dates as much values') 
    
    try: 
        values = jnp.array(values, dtype='float64') 
    except: raise ValueError

    #функция которая будет переводить пивот поинтс в интеджерс 

    #магия которая будет делать массив длинной в calendar 

    _ = [] 
    curve = jnp.array(_, dtype='float64')
    return curve

class Swap: 

    default_notional_value = 100 
    market_state = None

    def __init__(self, notional, start_date, end_date: Union[int, date], **kwargs): 

        if notional is None: 
            self.notional = Swap.default_notional_value
        
        if start_date is None: 
            self.start_date = 0 #equivalent to today

        if isinstance(start_date, date): 
            self.start_date = start_date - market_state.today_ #ensuring both start and end date are of int (in index sense) format

        if isinstance(end_date, date): 
            self.end_date = end_date - market_state.today_
        else: self.end_date = end_date

        #allows overriding market state 
        self.market_state = kwargs.get('market_state', Swap.market_state) #update market_state internally if needed
        self.market_state = kwargs.get('ms', Swap.market_state)

    @classmethod
    def update_market_state(cls, market_state): 
        cls.market_state = market_state

class OIS(Swap):
    '''Stores the quotes and provides cost calculations for the OIS contracts. (IN WORK)'''
    def __init__(self, notional, start_date, end_date: Union[int, date], fix_rate, **kwargs): 
        super().__init__(notional, start_date, end_date, **kwargs)
        self.fixed_rate = fix_rate


if __name__ == '__main__':

    @Testing.timer
    def calendars():
        '''requires pickling'''
        dates = Dates.custom_space(date(2025, 9 ,19), 122) 
        dates_np = np.array(dates, dtype='datetime64[D]')
        _ = jnp.array(dates_np, dtype='int32')
        dates_np_int = _ - (_[0]) 
        print(dates_np_int)

        fy = [0]*len(dates)
        
        #fraction of year in jbp format 
        for y in range(1, len(dates)): 
            fy[y] = Dates.frac_year(dates[y-1], dates[y], 1)
        fy = jnp.array(fy, dtype='float32')
        fy_cum = fy.cumsum()

        return dates, dates_np_int, fy, fy_cum
    
    dates, dates_np_int, fy, fy_cum = calendars()

    #termination dates in the datetime.date format 
    _ = list(Testing.termination_dates_dict.values())
    termination_dates = []
    for y in _:
        termination_dates.append(datetime.strptime(y, '%Y-%m-%d').date()) 

    #get the termination dates in jnp format 
    @Testing.timer
    def termination_dates_as_integer(termination_dates, dates, dates_np_int):
        _ = []
        for date in termination_dates: 
            if date in dates:
                _.append(int(dates_np_int[dates.index(date)]))
            else: raise ValueError(f'no such date {date} in the calendar')
        termination_dates_int = jnp.array(_, dtype='int16')

        return termination_dates_int
    
    term_dates = termination_dates_as_integer(termination_dates, dates, dates_np_int)
    
    ms = Market(date(2025, 9 ,19), 0.17, 11.2, term_dates)





    

    


        




        


