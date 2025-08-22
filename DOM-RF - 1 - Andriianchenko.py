import numpy as np
from datetime import datetime, timedelta

"""
Логика следующая:
1. Парсинг кривой доходности (parse_yeild_curve)
2. Проверка корректности введения дат (bond_pricing)
3. Определения дат, в которые совершаются выплаты по купонам (payments_periodicity)
ВАЖНО отметить, что допускается предпосылка, что они совершаются не позднее 28-го числа в любом месяце
4. Определения дат предыдущей и следующей даты выплаты купонов и расчёт НКД (bond_pricing)
5. Интерполяция ставки доходности бонда из КБД (bond_pricing)
6. PV вычисление (bond_pricing)
7. Функция (bond_pricing) возвращает искомую цену (price) и вычесленные по ходу её работы значения (other)
"""

"""
Следует улучшить:
1. Condition checking для бОльшего числа параметров
2. Раздробить функцию bond_pricing: accrued_interest и интерполяцию, можно сделать отдельно + duration
3. сделать Класс Bond с методами, которые представлены как функции сейчас
2. Интерфейизация 
3. Документация 
4. Подробные комментарии
"""

yield_curve_dict = {
    '1W': 7.49,
    '2W': 7.49,
    '1M': 7.48,
    '2M': 7.49,
    '3M': 7.52,
    '6M': 7.63,
    '9M': 7.73,
    '1Y': 7.82,
    '2Y': 8.32,
    '3Y': 8.65,
    '4Y': 8.85,
    '5Y': 9.01,
    '6Y': 9.16,
    '7Y': 9.36,
    '8Y': 9.60,
    '9Y': 9.86,
    '10Y': 10.13
}

def parse_yield_curve(yield_curve_dict):
    """
    yield_curve_dict to {float_years: decimal_yield}
    """
    yc = {}
    for k, v in yield_curve_dict.items():
        n = int(k[:-1])
        if k.endswith('W'):
            yrs = n / 52
        elif k.endswith('M'):
            yrs = n / 12
        else: 
            yrs = n
        yc[yrs] = v / 100
    return dict(sorted(yc.items()))

yield_curve = parse_yield_curve(yield_curve_dict)

def payments_periodicity(t0, T, frequency, maturity):
    '''
    t0, T - datetime format
    frequency (int)
    maturity (int) - years
    '''

    if not isinstance(t0, datetime) or not isinstance(T, datetime):
        raise ValueError("invalid inputs")
    
    if frequency <= 0 or frequency > 12 or 12 % frequency != 0:
        raise ValueError("invalid inputs")
    
    months_per_period = 12 // frequency

    year, month, day = t0.year, t0.month, t0.day

    if day >= 29: #that restriction is for the simplicity of the code
        raise ValueError("invalid inputs")

    max_payments = int(frequency * maturity) + 1 
    payment_dates = []

    for i in range(max_payments): 
        if month + months_per_period > 12:
            year += 1
            month = (month + months_per_period) - 12
        else:
            month += months_per_period

        payment_date = datetime(year, month, day)
        payment_dates.append(payment_date)

        if payment_date >= T: 
            break

    return payment_dates

def bond_pricing(release_date, current_date, FV=1000, C=0.05, periodicity=2, maturity=5, credit_spread=0.01): 
    '''
    FV - face value of the bond
    C - coupon rate
    periodicity - # of times bond pays coupons 
    maturity - # of years before the bond matures
    release_date - the day have been emited DD-MM-YYYY
    current_date - the day of purchase
    '''
    global yield_curve

    try: 
        t0 = datetime.strptime(release_date, "%d-%m-%Y")
        t1 = datetime.strptime(current_date, "%d-%m-%Y")
    except ValueError as e: 
        raise ValueError("unproper date format")
    
    T = t0.replace(year=(t0.year + maturity)) 
    
    if (t0 >= t1) or (t1 >= T): 
        raise ValueError("date order is incorrect")
    
    coupon_payment = (C*FV) / periodicity

    coupon_dates = payments_periodicity(t0, T, periodicity, maturity)

    previous_coupon = None
    next_coupon = None

    for coupon_date in coupon_dates:
        if coupon_date < t1:
            previous_coupon = coupon_date
        elif coupon_date >= t1 and next_coupon is None:
            next_coupon = coupon_date
            break

    #accrued interest determination
    if previous_coupon is None:
        days_since_last_coupon = (t1 - t0).days
        if next_coupon is not None:
            days_in_coupon_period = (next_coupon - t0).days
        else:
            days_in_coupon_period = (T - t0).days
    else:
        days_since_last_coupon = (t1 - previous_coupon).days
        if next_coupon is not None:
            days_in_coupon_period = (next_coupon - previous_coupon).days
        else:
            days_in_coupon_period = (T - previous_coupon).days

    accrued_interest = coupon_payment * (days_since_last_coupon / days_in_coupon_period)

    remaining_years = (T - t1).days / 365

    #ytm interpolation 
    keys = list(yield_curve.keys())
    for i in range(len(keys)-1):
        if keys[i] <= remaining_years <= keys[i+1]:
            w = (remaining_years - keys[i]) / (keys[i+1] - keys[i])
            ytm = yield_curve[keys[i]] * (1-w) + yield_curve[keys[i+1]] * w
            break

    df = ytm + credit_spread

    pv_coupons = 0.0
    for d in coupon_dates:
        if d > t1:
            t = (d - t1).days / 365
            pv_coupons += coupon_payment / ((1 + df/periodicity) ** (t * periodicity))

    pv_principal = FV / ((1 + df/periodicity) ** (remaining_years * periodicity))

    price = pv_principal + pv_coupons + accrued_interest #dirty price
    other = (df, ytm, remaining_years, accrued_interest)

    #ДЮРАЦИЯ
    cumsum = 0.0
    for d in coupon_dates:
        if d > t1:
            t = (d - t1).days / 365
            cumsum += coupon_payment * t / ((1 + df/periodicity) ** (t * periodicity))
    cumsum += FV / ((1 + df/periodicity) ** (remaining_years * periodicity)) * remaining_years
    duration = cumsum/price

    return [price, duration], other

if __name__ == "__main__": 
    result, _ = bond_pricing(
        release_date="01-01-2024", 
        current_date="01-07-2024"
    )
    print(result)
    print(_)