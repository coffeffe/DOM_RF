import numpy as np
"""
Логика следующая:
1. Парсинг кривых доходности (parse_yeild_curve)
2. Интерполяция кривых доходности, чтобы получить ставки. (interpolate_yield_curve)
В отличие от задания 1, абстрактая репрезентация данных в этом задании позволяет определить ставки 1к1 из кривой ставок, поэтому тут модифицирован фрагмент из Задания 1
3. Расчёт форвардных ставок (forward_rates)
Если я правильно воспринял: "будет выплачивать среднеарифметическую квартальную плавающую ставку"
В контексте нашей задачи это превращая просто в доступные квартальные ставки по рублю, которые можно аппроксимировать из кривой ставок
4. Расчёт ставок дисконтирования (discount_factors) - вспомогательная
5. Высчитываем текующую стоимость обоих поток + арифметика = выходит эксплицитное выражение для фиксированной ставки (calculate_fair_fixed_rate)
"""

"""
Следует улучшить:
1. Верификация фиксированной ставки (в смысле, что она реально сравнивает два дисконтированных потока)
2. Интерполяцию можно? реализовать через numpy (?) 
3. проверка валидности введённых данных 
4. документация 
5. Подробные комментарии
6. Может присутствовать ошибка в логике -> скрипт станет не пригодным полностью
"""

yield_curve_rub_dict = {
    '1W': 15.90, '2W': 16.00, '1M': 16.00, '2M': 16.00, '3M': 15.94,
    '6M': 15.65, '9M': 15.24, '1Y': 14.97, '2Y': 14.16, '3Y': 14.27,
    '4Y': 14.72, '5Y': 15.33
    }
    
yield_curve_cny_dict = {
    '1W': 4.49, '2W': 5.23, '1M': 5.24, '2M': 5.48, '3M': 7.17,
    '6M': 7.09, '9M': 7.24, '1Y': 7.16, '2Y': 7.35, '3Y': 7.79,
    '4Y': 8.27, '5Y': 8.75
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

#илд кёрвы вынесены за класс Своп_контракт, потому что 1. легче стресс тесты. 2. Это макрономические показатели, которые будет легко менять в контексте одного контакрта 
yc_rub = parse_yield_curve(yield_curve_rub_dict)
yc_cny = parse_yield_curve(yield_curve_cny_dict)

class Swap_contract():

    def __init__(self, notional, spot_rate, maturity, periodicity): 
        '''
        notional - номинал (по умолчанию в рублях)
        spot_rate - e.g. CNY/RUB
        maturity - in years
        periodicity - # of payments per year
        '''
        self.notional = notional
        self.notional_foreign = notional / spot_rate
        self.spot_rate = spot_rate
        self.maturity = maturity
        self.periodicity = periodicity
        #assume t0 = 0, T = maturity (in years)
        self.payment_dates = np.linspace(1/periodicity, maturity, periodicity*maturity)

    def interpolate_yield_curve(self, yield_curve):
        keys = sorted(yield_curve.keys())
        interpolated_yields = []
        
        for tick in self.payment_dates:
            if tick in keys:
                interpolated_yields.append(yield_curve[tick])
            else: 
                for i in range(len(keys)-1):
                    if keys[i] <= tick <= keys[i+1]:
                        w = (tick - keys[i]) / (keys[i+1] - keys[i])
                        ytm = yield_curve[keys[i]] * (1-w) + yield_curve[keys[i+1]] * w
                        interpolated_yields.append(ytm)
                        break

        return interpolated_yields
    
    def forward_rates(self, yield_curve):
        '''
        Мы верим в теорию ожиданий, поэтому утверждаем, что форвардные ставки соответсвуют будущим спотовым. 
        (1 + f1)(1 + f1_2) = (1+f2)^2, где fx - годовые ставви за год х, и fx_y, форвардная ставка с года х по н
        '''
        yields = self.interpolate_yield_curve(yield_curve)
        frwd_rates = []

        for i in range(len(yields)): 
            if i == 0: 
                frwd_rates.append(yields[i])
                continue
            temp1 = (1 + yields[i])**(self.payment_dates[i])
            temp2 = (1 + yields[i-1])**(self.payment_dates[i-1])
            temp12 = temp1/temp2
            temp = temp12**(1/(self.payment_dates[i]-self.payment_dates[i-1]))
            frwd_rate = temp - 1
            frwd_rates.append(frwd_rate)
        
        return frwd_rates
    
    def discount_factors(self, yield_curve):
        yields = self.interpolate_yield_curve(yield_curve)
        discount_factors = []
        
        for t, y in zip(self.payment_dates, yields):
            df = 1 / (1 + y)**t
            discount_factors.append(df)
            
        return discount_factors

    def calculate_fair_fixed_rate(self, rub_curve, cny_curve):
        """
        В вложении в письме на почте, будет примерно расписана логика, которая примяется здесь. 
        """
        rub_forwards = self.forward_rates(rub_curve)
        rub_dfs = self.discount_factors(rub_curve)
        
        # PV of cashflows on russian nominal
        floating_pv = 0
        period_fraction = 1 / self.periodicity
        
        for fwd_rate, df in zip(rub_forwards, rub_dfs):
            payment = self.notional * fwd_rate * period_fraction
            pv = payment * df
            floating_pv += pv
        
        cny_dfs = self.discount_factors(cny_curve)
        
        # Explanation on the paper in the mail 
        fixed_rate = floating_pv * self.periodicity / (self.notional_foreign * self.spot_rate * sum(cny_dfs))
        
        return fixed_rate

    

if __name__ == "__main__": 

    swap_rub_cny = Swap_contract(
        notional=1_000_000_000,  
        spot_rate=12.8,          
        maturity=1,              
        periodicity=4            
    )
    
    fair_rate = swap_rub_cny.calculate_fair_fixed_rate(yc_rub, yc_cny)
    print(fair_rate)
