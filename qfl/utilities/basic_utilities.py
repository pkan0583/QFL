"""
Created on Sun Jan 03 13:02:07 2016
@author: Benn
"""
import datetime as dt
import pandas as pd
import numpy as np
import json
from workalendar.europe import EuropeanCentralBank, Germany, UnitedKingdom, \
    France, Sweden, Switzerland, Spain, Italy
from workalendar.america import Brazil, Mexico, Chile
from workalendar.usa import UnitedStates
from workalendar.asia import SouthKorea, Japan, Taiwan
from workalendar.oceania import Australia
from pandas.tseries.offsets import CustomBusinessDay
import qfl.core.constants as constants
import logging
from logging.config import dictConfig

logging_config = (dict(
    version=1,
    formatters={
        'f': {'format':
              '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
        },
    handlers={
        'h': {'class': 'logging.StreamHandler',
              'formatter': 'f',
              'level': logging.INFO}
        },
    root={
        'handlers': ['h'],
        'level': logging.INFO,
        },
))


def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def is_iterable(x=None):

    y = getattr(x, "__iter__", None)
    if y is None:
        return False
    else:
        return True


def is_date_type(x=None):

    if isinstance(x, dt.datetime) or isinstance(x, dt.date) \
            or isinstance(x, pd.Timestamp) or isinstance(x, np.datetime64):
        return True
    else:
        return False


def replace_nan_with_none(df=None):
    df = df.where((pd.notnull(df)), None)
    return df


def json_column_from_columns(df=None, columns=None, new_col_name=None):

    df = df.copy(deep=True)
    for column in df.columns:
        if is_date_type(df[column].values[0]):
            df[column] = df[column].astype('str')

    tmp = df[columns].to_dict(orient='records')
    tmp = [json.dumps(k) for k in tmp]
    df[new_col_name] = tmp

    for col in columns:
        del df[col]

    return df


def df_columns_from_json_column(df=None, json_col_name=None):

    df = df.copy(deep=True)
    new_cols = [json.loads(d) for d in df[json_col_name].values]
    new_cols = pd.DataFrame.from_dict(new_cols)
    new_cols.index = df.index
    df[new_cols.columns] = new_cols
    del df[json_col_name]

    return df


def numeric_cap_floor(x=None, cap=None, floor=None):

    if isinstance(x, pd.Series):
        x = x.clip(upper=cap, lower=floor)
    elif isinstance(x, np.ndarray):
        x = x.clip(min=floor, max=cap)
    else:
        if cap is not None:
            x = min(x, cap)
        if floor is not None:
            x = max(x, floor)
    return x

"""
-------------------------------------------------------------------------------
BUSINESS DATE AND CALENDAR UTILITIES
-------------------------------------------------------------------------------
"""


def get_futures_month_from_code(month_codes=None):

    if isinstance(month_codes, str):
        futures_months = constants.futures_month_codes\
                             .values().index(month_codes) + 1
    else:
        futures_months = [get_futures_month_from_code(code)
                          for code in month_codes]
    return futures_months


class DateUtils(object):

    default_calendar = 'UnitedStates'
    default_calendar_start_year = 1970
    default_calendar_end_year = 2050

    # Dictionary of string to workalendar.calendar
    # Some of these are approximations eg NZFE, SGX, SHFE
    exchange_calendar_map = {
        'US': 'UnitedStates',
        'UK': 'UnitedKingdom',
        'GR': 'Germany',
        'EUREX': 'EuropeanCentralBank',
        'CBOE': 'UnitedStates',
        'CME': 'UnitedStates',
        'ICE': 'UnitedStates',
        'SGX': 'Taiwan',
        'SHFE': 'Taiwan',
        'LIFFE': 'UnitedKingdom',
        'NZFE': 'Australia'
    }

    # Dictionary of string to workalendar.calendar
    calendar_map = {
        'EuropeanCentralBank': EuropeanCentralBank,
        'Germany': Germany,
        'UnitedKingdom': UnitedKingdom,
        'France': France,
        'Sweden': Sweden,
        'Switzerland': Switzerland,
        'Spain': Spain,
        'Italy': Italy,
        'Brazil': Brazil,
        'Mexico': Mexico,
        'Chile': Chile,
        'SouthKorea': SouthKorea,
        'Japan': Japan,
        'Taiwan': Taiwan,
        'Australia': Australia,
        'UnitedStates': UnitedStates
    }

    # This is a dictionary of string to CustomBusinessDay
    custom_bdays = None

    # Pre-initialize calendars upon loaading
    @classmethod
    def initialize(cls):
        cls.custom_bdays = dict()
        for cal in cls.calendar_map:
            cls.custom_bdays[cal] = CustomBusinessDay(
                holidays=cls.get_holidays(cal).keys())

    @classmethod
    def get_bday(cls, calendar_name):
        if cls.custom_bdays is None:
            cls.initialize()
        if calendar_name in cls.calendar_map:
            return cls.custom_bdays[calendar_name]
        else:
            raise LookupError('calendar not supported!')

    @classmethod
    def get_holidays(cls,
                     calendar=default_calendar,
                     start_year=default_calendar_start_year,
                     end_year=default_calendar_end_year):

        holiday_list = dict()
        holiday_calendar = cls.calendar_map[calendar]()
        for year in range(start_year, end_year):
            holiday_list.update(dict(holiday_calendar.holidays(year)))
        return holiday_list

    @classmethod
    def get_business_date_range(cls,
                                start_date=None,
                                end_date=dt.datetime.today(),
                                calendar_name=default_calendar):
        bday = cls.get_bday(calendar_name)
        dates = pd.DataFrame(pd.date_range(start_date, end_date))
        dates.index = dates[0]
        dates = dates.asfreq(bday).index
        return dates

    @classmethod
    def is_business_day(cls,
                        date=None,
                        calendar_name=default_calendar):

        my_bday = cls.get_bday(calendar_name)
        test_bday = date + my_bday - my_bday
        if test_bday == date:
            return True
        else:
            return False

    @classmethod
    def closest_business_day(cls,
                             date=None,
                             prev=True,
                             calendar_name=default_calendar):

        my_bday = cls.get_bday(calendar_name)

        if date is None:
            date = dt.datetime.today()
        if prev:
            return_date = date + my_bday - my_bday
        else:
            return_date = date - my_bday + my_bday
        return return_date

    @classmethod
    def workday(cls,
                date=None,
                num_days=1,
                calendar_name=default_calendar):

        my_bday = cls.get_bday(calendar_name)

        if date is None:
            date = dt.datetime.today()

        return date + (num_days * my_bday)

    @classmethod
    def networkdays(cls,
                    start_date=None,
                    end_date=None,
                    calendar_name=default_calendar):

        if (isinstance(start_date, pd.Series) and isinstance(end_date, pd.Series))\
                or (isinstance(start_date, pd.tseries.index.DatetimeIndex)
                and isinstance(end_date, pd.tseries.index.DatetimeIndex)):
            hol = cls.get_holidays(calendar_name).keys()
            A = [d.date() for d in start_date]
            B = [d.date() for d in end_date]
            return np.busday_count(A, B, holidays=hol)

        if is_iterable(start_date):
            if len(start_date) == 0:
                raise ValueError("start date cannot be zero-length!")

        is_series = False
        if isinstance(start_date, pd.Series):
            is_series = True
            orig_index = start_date.index
            start_date = start_date.values

        if isinstance(end_date, pd.Series):
            end_date = end_date.values

        my_bday = cls.get_bday(calendar_name)

        if is_iterable(start_date):
            df = pd.DataFrame(data=np.array(start_date))
            df = df.rename(columns={0: 'start_date'})
            df['end_date'] = end_date
        else:
            df = pd.DataFrame(index=[0], columns=['start_date',
                                                  'end_date',
                                                  'net_workdays'])
            df.loc[0, 'start_date'] = start_date
            df.loc[0, 'end_date'] = end_date

        df['net_workdays'] = np.nan
        min_date = df['start_date'].min()
        max_date = df['end_date'].max()

        dates = pd.DatetimeIndex(start=min_date, end=max_date, freq=my_bday)
        dates = pd.DataFrame(index=dates)

        for i in range(0, len(df)):
            df.loc[i, 'net_workdays'] = len(dates[df.loc[i, 'start_date']
                                                 :df.loc[i, 'end_date']])-1

        if is_series:
            return pd.Series(index=orig_index,
                             data=df['net_workdays'].values)
        elif len(df) == 1:
            return df['net_workdays'].values[0].tolist()
        else:
            return df['net_workdays'].values.tolist()


def get_days_from_maturity(start_date=None,
                           maturity_date=None,
                           date=None,
                           calendar_name='UnitedStates'):

        total_days = networkdays(start_date=start_date,
                                 end_date=maturity_date,
                                 calendar_name=calendar_name)
        days_elapsed = networkdays(start_date=start_date,
                                   end_date=date,
                                   calendar_name=calendar_name)
        days_to_maturity = total_days - days_elapsed

        # Handle forward start
        if is_iterable(days_elapsed):
            days_elapsed[days_elapsed < 0] = 0
        else:
            if days_elapsed < 0:
                days_elapsed = 0

        return days_to_maturity, days_elapsed, total_days


def networkdays(start_date=None,
                end_date=None,
                calendar_name=DateUtils.default_calendar):

    return DateUtils.networkdays(start_date, end_date, calendar_name)


def calendarday(date=None, num_days=1):
    if isinstance(num_days, pd.DataFrame) or isinstance(num_days, pd.Series):
        return date + pd.TimedeltaIndex(num_days.values, unit='D')
    else:
        return date + dt.timedelta(days=num_days)


def workday(date=None,
            num_days=1,
            calendar_name=DateUtils.default_calendar):
    return DateUtils.workday(date, num_days, calendar_name)


def closest_business_day(date=None,
                         prev=True,
                         calendar_name=DateUtils.default_calendar):

    return DateUtils.closest_business_day(date, prev, calendar_name)


def is_after_time(hour=None, minute=None):

    t = dt.time(hour, minute)
    return dt.datetime.today() > t
