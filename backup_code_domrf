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
import matplotlib.pyplot as plt 
import pickle

#to see unfinished code type in "IN WORK"

'''
List of problems: 
1. Initialization and parsing of quotes - resolve as data vase access is present
2. Do termination dates for each maturity coincide across every type of contract traded (СПФИ)? 
    2.1. If yes: create separate variable for the collection of tenors and termination dates
3. Quote logic (its absence)
4. Solver for curve. How to make it fast
5. Calendar class added Market state class is kinda fuzzy ngl fr fr 
6. MB we should make da MEGAINSTANCE for the 4 instances of Calendar (??) 
'''

'''
To-do list: 
1. Interpolation
2. curve() into Swap integration
3. docstrings (SWAPS)
4. Solver for curve. How to make it fast
'''

class Solver: 
    '''(IN WORK)'''
    pass

@dataclass
class Curves: 
    '''(IN WORK)'''
    pivot_points: Union[list[date], jax.Array]
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
        termination_dates_datetime(): Returns datetime.date format list of the termination dates as at 19.09.2025
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
    
    @staticmethod
    def termination_dates_datetime(**kwargs):
        '''Returns datetime.date format list of the termination dates as at 19.09.2025'''
        termination_dates_dictionary = kwargs.get('term_dates', Testing.termination_dates_dict)
        termination_dates_dictionary = kwargs.get('termination_dates', Testing.termination_dates_dict)
        termination_dates_dictionary = kwargs.get('term_dates_dict', Testing.termination_dates_dict)
        termination_dates_dictionary = kwargs.get('termination_dates_dict', Testing.termination_dates_dict)

        #termination dates in the datetime.date format 
        _ = list(termination_dates_dictionary.values())
        termination_dates = []
        for y in _:
            termination_dates.append(datetime.strptime(y, '%Y-%m-%d').date())

        return termination_dates

class Dates: 
    '''
    Basic operation with dates (IN WORK)

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
            convention: 0 or "FOLLOWING" - following convention ensures the date is moved to the closest working day \n 1 or "MODFOLLOWING" -  following convention ensures the date is moved to the closest working day within a month

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
        '''Returns dictionary of the terminaton dates for the standard maturities based on the start date.'''
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
    '''Class is designed to store calendars and methods manipulating these calendars. (IN WORK)
    
    !!! Consider merging with Dates class 

    Parameters: 
        today_ (date): default parameter. Literally the current day (today)
        start_date (date): parameter shifting the day zero for the simulation/pricing. Substitutes a value of today_. 

    Public methods: 
        calendars(): Generates 4 intances: dates in datetime.date, dates as integer, 
                     fraction of year for each incremental day and cummulative fractoin of year
        projection_of_dates_as_integer(): projects and array of datetime.date variables into jax.Array in accordance with the calendar of integer form.
        gorinich(): consider deliting the function for its obsolesence 
        save_calendars(): pickles the calendars generated by calendars() function for later use
        get_calendars(): loads the pickled before calendars. Shifts the day-zero in the calendars if needed. (IN WORK)
        payment_dates(): bbtain the payment dates from termination dates in accrodance with the business day offset by 1D and FOLLOWING convention
    '''
    today_ = datetime.today().date()

    def __init__(self, **kwargs):
        '''(!!!)'''
        start_date = kwargs.get('start_date', Calendar.today_)
        pass
    
    @Testing.timer
    @staticmethod
    def calendars(start_date=None, **kwargs):
        '''Generates 4 intances: dates in datetime.date, dates as integer, fraction of year for each incremental day and cummulative fractoin of year
        
        Args:
            start_date (datetime.date): day zero for the calendar instances
            **kwords: additional parameters

        Keyword Args: 
            span_months (int): desired length of the calendars in month. Default is 122 months 

        Returns:
            dates (list[dates]): calendar of list type where each element is of datetime.date format 
            dates_np_int (jax.Array): calendar of jax array format, each day is integer starting with 0 and incrementing by 1
            fy (jax.Array): fraction of year for each day_{t} - day_{t-1} interval 
            fy_cum (jax.Array): cummulative fraction of year array
        '''
        if start_date is None: 
            start_date = Calendar.today_
        span_months = kwargs.get('span_months', 122)

        dates = Dates.custom_space(start_date, span_months) 
        dates_np = np.array(dates, dtype='datetime64[D]')
        _ = jnp.array(dates_np, dtype='int32')
        dates_np_int = _ - (_[0]) 
        print(dates_np_int)

        fy = [0]*len(dates)
        
        #fraction of year in jnp format 
        for y in range(1, len(dates)): 
            fy[y] = Dates.frac_year(dates[y-1], dates[y], 1)
        fy = jnp.array(fy, dtype='float32')
        fy_cum = fy.cumsum()

        return dates, dates_np_int, fy, fy_cum
    
    @Testing.timer
    def projection_of_dates_as_integer(projected_dates, dates, dates_np_int):
        '''Projects and array of datetime.date variables into jax.Array in accordance with the calendar of integer form. 
        
        Args: 
            projected_dates (list[date]): list of dates to be projected into integer
            dates (list[date]): calendar in space of which the dates are projected
            dates_np_int (jax.Array): calendar in space of which the dates are projected in integer representation
            
        Returns: 
            jax.Array of integer values of projected original dates
            
        Example: 
            >>>x = projection_of_dates_as_integer(date(2025, 1, 2), 
                                                 [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)],
                                                  jnp.Array(0, 1 ,2))
            >>>x
            Array(2, dtype='int8')
        '''
        _ = []
        for date in projected_dates: 
            if date in dates:
                _.append(int(dates_np_int[dates.index(date)]))
            else: raise ValueError(f'no such date {date} in the calendar')
        projected_dates_int = jnp.array(_, dtype='int16')

        return projected_dates_int

    @staticmethod
    def gorinich(calendar): 
        '''
        !!! PROTOTYPE. CONSIDER DELETE'''
        calendar_np = np.array(calendar, dtype='datetime64[D]')
        _ = jnp.array(dates_np, dtype='int')
        calendar_int = _ - (_[0] - 1) 
        return calendar, calendar_np, calendar_int 

    @staticmethod
    def save_calendars(**kwargs): 
        ''' Pickles the calendars generated by calendars() function for later use. 

        Args: 
            **kwords: the 4 calendar instances or none

        Keyword Args: 
            dates (list[dates]): calendar of list type where each element is of datetime.date format 
            dates_np_int (jax.Array): calendar of jax array format, each day is integer starting with 0 and incrementing by 1
            fy (jax.Array): fraction of year for each day_{t} - day_{t-1} interval 
            fy_cum (jax.Array): cummulative fraction of year array

        Returns: 
            None
        '''
        dates = kwargs.get('dates', None)
        dates = kwargs.get('dates', None)
        dates_np_int = kwargs.get('dates_np_int', None)
        fraction_of_year = kwargs.get('fraction_of_year', None)
        fraction_of_year = kwargs.get('fy', None)
        fraction_of_year_cum = kwargs.get('fraction_of_year_cum', None)
        fraction_of_year_cum = kwargs.get('fy_cum', None)

        _ = {'dates': dates, 'dates_np_int': dates_np_int, 'fraction_of_year': fraction_of_year, 
             'fraction_of_year_cum': fraction_of_year_cum}
        
        for name, array in _.items(): 
            if array is not None: 
                picklename = f'{name}.pkl'
                with open(picklename, 'wb') as f: 
                    pickle.dump(array, f)  

        return None 

    @Testing.timer
    @staticmethod
    def get_calendars(**kwargs):
        '''Loads the pickled before calendars. Shifts the day-zero in the calendars if needed. (IN WORK)

        !!! Add ability to handle type(int)

        Keyword Args: 
            start_date (datetime.date): Allows to shift the day-zero of calendars to the desired one. 

        Returns: 
            None or 
            dates (list[dates]): calendar of list type where each element is of datetime.date format 
            dates_np_int (jax.Array): calendar of jax array format, each day is integer starting with 0 and incrementing by 1
            fy (jax.Array): fraction of year for each day_{t} - day_{t-1} interval 
            fy_cum (jax.Array): cummulative fraction of year array            
        '''
        #load pickles back into the system
        start_date = kwargs.get('start_date', Calendar.today_)

        loaded_objects = []
        for filename in ['dates.pkl', 'dates_np_int.pkl', 'fraction_of_year.pkl', 'fraction_of_year_cum.pkl']: 
            if os.path.isfile(filename): 
                with open(filename, 'rb') as f: 
                    array = pickle.load(f)
                    loaded_objects.append(array)
            else: return "No calendar files in the directory"

        _dates, _dates_np_int, _fraction_of_year, _fraction_of_year_cum = loaded_objects

        if isinstance(start_date, date): 

            #check if the start date is in the calendar
            if start_date < _dates[0]: 
                print(start_date)
                raise ValueError('Unavailable start_date parameter')
            
            if start_date == _dates[0]: 
                print('start date coincides with the initial date in the calendar')
                return _dates, _dates_np_int, _fraction_of_year, _fraction_of_year_cum #TODO
            
            print('start date is greater then the initial date in the calendar')
            delta = (start_date - _dates[0]).days
            
            dates = _dates[delta:]
            dates_np_int = _dates_np_int[delta:] - delta
            _fraction_of_year = _fraction_of_year[delta:]
            fraction_of_year = _fraction_of_year.at[0].set(0.0)
            _fraction_of_year_cum = _fraction_of_year_cum[delta:]
            fraction_of_year_cum = _fraction_of_year_cum.at[0].set(0.0)

            return dates, dates_np_int, fraction_of_year, fraction_of_year_cum
        
        if isinstance(start_date, int): #TODO
            pass

        return None
    
    @staticmethod
    def payment_dates(termination_dates: Union[list, jax.Array], **kwargs): 
        '''Obtain the payment_dates in accrodance with the business day offset by 1D and FOLLOWING convention 
        as declared in the standard swap contracts. That includes IRS, OIS, XCCY. Handles both list of dates and jnp.array
        
        Args:
            termination_dates (): termination dates of the swaps (assumed to be the same for every swap: irs, ois, XXCY)
            start_date (date): optional arguments. It is used in cases if termination_dates are inputed in integer format
            
        Returns: 
            payments dates corresponding to each termination dates'''
        if isinstance(termination_dates, jax.Array): 
            start_date = kwargs.get('start_date', None)
            if start_date is None: 
                raise ValueError('Can not determine the payment dates as there is no regerence starting date')

            _ = []
            for date_jax in termination_dates: 
                date_ = int(date_jax)
                _2 = start_date + timedelta(days=date_) + timedelta(days=1)
                _.append(_2)
            
            payment_dates = Dates.business_day(_, 0)
            return payment_dates

        return None

        
@dataclass
class Market: 
    '''
    Stores current market parameters used at a wide range of calculations. (IN WORK)

    Public methods: 
        update(): updates market paramaters in accordance with the current state. (IN WORK)

    Parameters: 
         today_ (datetime.date): current date (the literal today or the desired one)
         key_rate (float): current CB key rate 
         rub_cny_spot (float): rub/cny spot rate at the current date
         termination_dates (list[dates] or jax.Array): termination dates outstanding for the swaps (should be complient with today_)
    Calendar parameters: 
        dates (list[dates]): calendar of list type where each element is of datetime.date format 
        dates_np_int (jax.Array): calendar of jax array format, each day is integer starting with 0 and incrementing by 1
        fy (jax.Array): fraction of year for each day_{t} - day_{t-1} interval 
        fy_cum (jax.Array): cummulative fraction of year array          
    '''
    def __init__(self, key_rate: float, rub_cny_spot: float, **kwargs):
        self.today_: date = kwargs.get('today', Calendar.today_)
        self.today_: date = kwargs.get('today_date', Calendar.today_)
        self.today_: date = kwargs.get('start_date', Calendar.today_)
        self.key_rate = key_rate 
        self.rub_cny_spot = rub_cny_spot
        self.termination_dates = kwargs.get('termination_dates', None)

        self.dates, self.dates_np_int, self.fraction_of_year, self.fraction_of_year_cum = Calendar.get_calendars(start_date=self.today_)

    def update(self): 
        '''
        Updates market paramaters in accordance with the current state. (IN WORK)

        !!!Requires addition of parsing values for rub/sny spot rate and key rate. 
        '''

        self.today_ = date.today()
        #self.ley_rate_0 = 
        #self.rub_cny_spot = 
        #figure out a way to parse the values from a source trusted by domrf firewall

class Quote: 
    pass

@Testing.timer
def discounting_series(float_rate_series): 
    '''Generates series of discounting factor. The functoin is coupled with floating_rate_series. 
    Every discount factor corresponds to fraction of year value, hence it correspongs to calendar (dates).
    
    Args: 
        float_rate_series (jax.Array): product of float_rate_series()
        
    Returns: series of discounting factors
    '''
    _float_rate_serioes_wo_first = float_rate_series[1:] #to ensure nothing is devided by zero as it is the first element 
    _ = _float_rate_serioes_wo_first.cumprod()
    _ = 1 / _
    discounting_series = jnp.insert(_, 0, 1.0)
    return discounting_series

@Testing.timer
def discount_factor(curve, fraction_of_year_dates, zero_date, date_):
    '''Get discount factor for specific date. (IN WORK)
    
    !!! Assess the need for such function. discounting_series() takes 0.15s on average 
    '''
    df = 0

    if isinstance(date_, date): 
        date_int = (date_ - zero_date).days()
    if isinstance(date_, int): 
        _ = date_
        date_ = zero_date + timedelta(days=date_)
        date_int = _ 

    if date_int == 0: 
        df = 1 
        return df
    
    float_rate_at_date = curve[date_int-1]*fraction_of_year_dates[date_int]

@Testing.timer
def float_rate_series(curve, fraction_of_year_dates): 
    '''Generates the flaoting rate in accordance with funding curve and current calendar (Calendar.dates)
    
    Args:
        curve (jax.Array): values corresponding to the funding rate at a particular date
        fraction_of_year_dates (jax.Array): fraction of year corresponding to day_{t}-day{t-1} interval

    Returns: floating rate series for the calendar
    '''
    _fraction_of_year_dates_wo_first = fraction_of_year_dates[1:]
    _curve_wo_last = curve[:-1]

    _ = jnp.multiply(_fraction_of_year_dates_wo_first, _curve_wo_last)
    _ = _ + 1
    float_rate_series = jnp.insert(_, 0, 0)

    return float_rate_series

@Testing.timer
def curve(pivot_points: Union[list[date], jax.Array], values: Union[list, jax.Array], calendar_length): 
    '''Generates the rate curve in accordance with the outstanding calendar. (IN WORK)
    
    Args: 
        pivot_points: the date for which the change of interest rate is assumed
        values: assumed values of the rate at the pivot points 
        calendar_length: the length of the calendar period desired
        
    Returns:
        Rate curve (non-interpolated) for the period desired'''
    if len(pivot_points) != len(values): 
        raise ValueError('There should be as much pivot dates as much values') 
    
    try: 
        _values = jnp.array(values, dtype='float32') 
    except: raise ValueError

    if isinstance(pivot_points, list): 
        try: 
            _pivot_points = Calendar.termination_dates_as_integer(pivot_points)
        except: raise ValueError
    else: _pivot_points = pivot_points

    #creating a simple list to turn it into jnp.array next 
    _ = [0]*calendar_length
    
    previous = 0
    for y, el in enumerate(_pivot_points): 
        date = int(el)
        _[previous:date] = [values[y]]*len(_[previous:date])
        previous = date

    #ensuring if the tail of the curve is non-zero: 
    _[previous:] = [_[previous-1]]*len(_[previous:])

    result = jnp.array(_, dtype='float32') 
    return result 

class Swap: 
    '''(IN WORK)'''
    default_notional_value = 100 
    market_state = None

    def __init__(self, end_date: Union[int, date], **kwargs): 
        '''(IN WORK)
        start_date is by default 0 
        '''
        self.notional = kwargs.get('notional', Swap.default_notional_value)

        start_date = kwargs.get('start_date', 0)

        if isinstance(start_date, date): 
            self.start_date = (start_date - Swap.market_state.today_).days() #ensuring both start and end date are of int (in index sense) format
            self.start_date_datetime = start_date
        else: 
            self.start_date = start_date
            self.start_date_datetime = Swap.market_state.dates[self.start_date]

        if isinstance(end_date, date): 
            self.end_date = (end_date - Swap.market_state.today_).days()
            self.end_date_datetime = end_date
        else: 
            self.end_date = end_date
            self.end_date_datetime = Swap.market_state.dates[self.end_date]

        #allows overriding market state 
        self.market_state = kwargs.get('market_state', Swap.market_state) #update market_state internally if needed
        self.market_state = kwargs.get('ms', Swap.market_state)

    @classmethod
    def update_market_state(cls, market_state): 
        cls.market_state = market_state

class OIS(Swap):
    '''Stores the quotes and provides cost calculations for the OIS contracts. (IN WORK)'''
    def __init__(self, end_date: Union[int, date], fix_rate, **kwargs): 
        super().__init__(end_date, **kwargs)
        self.fixed_rate = fix_rate

    def _payment_dates(self): 
        '''Consider moving the function into Calerndar class. OBSOLETE
        
        !!! consider deleting that function as it doubles the Calendar.payment_dates()
        '''
        if self.end_date - 366 <= self.start_date: 
            payment_dates = Calendar 
        
        

        return None

    def cost(self): 
        pass

    


if __name__ == '__main__':

    # dates, dates_np_int, fy, fy_cum = Calendar.get_calendars(start_date=date(2025,9,19))
    # calendar_length = len(dates_np_int)

    termination_dates = Testing.termination_dates_datetime()

    ms = Market(start_date=date(2025, 9 ,19), key_rate=0.17, rub_cny_spot=11.2)
    Swap.market_state = ms

    term_dates = Calendar.projection_of_dates_as_integer(termination_dates, ms.dates, ms.dates_np_int)

    ms.termination_dates = term_dates

    ruonia_implied = Testing.ois_ruonia_mid #just imagine

    calendar_length = len(ms.dates)
    ois_ruonia = curve(ms.termination_dates, ruonia_implied, calendar_length)

    # plt.plot(ois_ruonia)
    # plt.show()

    fr = float_rate_series(ois_ruonia, ms.fraction_of_year)

    df = discounting_series(fr)

    print(ms.fraction_of_year_cum)
    print(ms.fraction_of_year)

    payment_date_check = Calendar.payment_dates(termination_dates=term_dates, start_date=date(2025,9,19))
    print(payment_date_check) #works just as intended!!!!!!!!!11

    # ois_1w = OIS(end_date=ms.termination_dates[0], fix_rate=0.16)
    # print(ois_1w._payment_dates())


    










    

    


        




        


