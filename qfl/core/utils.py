import datetime as dt
import pandas as pd
from pandas.tseries.offsets import BDay, CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

default_calendar = USFederalHolidayCalendar()


def is_iterable(x=None):

    y = getattr(x, "__iter__", None)
    if y is None:
        return False
    else:
        return True


def is_business_day(date=None, calendar=default_calendar):

    my_bday = CustomBusinessDay(calendar=calendar)
    test_bday = date + my_bday - my_bday
    if test_bday == date:
        return True
    else:
        return False


def closest_business_day(date=None, prev=True, calendar=default_calendar):

    my_bday = CustomBusinessDay(calendar=calendar)

    if date is None:
        date = dt.datetime.today()
    if prev:
        return_date = date + my_bday - my_bday
    else:
        return_date = date - my_bday + my_bday
    return return_date


def workday(date=None, num_days=1, calendar=default_calendar):

    my_bday = CustomBusinessDay(calendar=calendar)

    if date is None:
        date = dt.datetime.today()

    return date + num_days * my_bday
