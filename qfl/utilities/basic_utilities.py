"""
Created on Sun Jan 03 13:02:07 2016
@author: Benn
"""
import datetime as dt
import pandas as pd
import numpy as np
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


def replace_nan_with_none(df=None):
    df = df.where((pd.notnull(df)), None)
    return df

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

        return date + num_days * my_bday

    @classmethod
    def networkdays(cls,
                    start_date=None,
                    end_date=None,
                    calendar_name=default_calendar):

        if isinstance(start_date, pd.Series):
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

        if len(df) == 1:
            return df['net_workdays'].values[0].tolist()
        else:
            return df['net_workdays'].values.tolist()


def networkdays(start_date=None,
                end_date=None,
                calendar_name=DateUtils.default_calendar):

    return DateUtils.networkdays(start_date, end_date, calendar_name)


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
