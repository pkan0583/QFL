
import pandas as pd
import datetime as dt
import pandas_datareader.data as pdata
import matplotlib.pyplot as plt
import numpy as np
import pyfolio
import urllib
import pymc
import requests
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
import statsmodels.tsa.stattools as sm
import qfl.core.data_interfaces as data_int
import qfl.core.database_interface as qfl_data
from qfl.core.data_interfaces import QuandlApi, YahooApi
from qfl.core.database_interface import DatabaseInterface as db
from qfl.core.database_interface import DatabaseUtilities as dbutils
import qfl.core.market_data as mkt_data
import qfl.macro.macro_models as macro
import qfl.etl.data_ingest as etl
import qfl.core.calcs as lib
import qfl.utilities.basic_utilities as utils
from scipy import interpolate
import matplotlib.ticker as mtick
from matplotlib import cm
import logging
from pandas.tseries.offsets import BDay

reload(etl)
reload(qfl_data)
reload(data_int)
reload(mkt_data)
from qfl.core.data_interfaces import QuandlApi, YahooApi, IBApi, FigiApi
from qfl.core.database_interface import DatabaseInterface as db, DatabaseUtilities as dbutils

db.initialize()

rebalance_frequency = 63

portfolio_target_weights = {
    'SPY': 0.10,
    'QQQ': 0.05,
    'IWM': 0.05,
    'EFA': 0.20,
    'EWU': 0.05,
    'EWJ': 0.05,
    'EEM': 0.10,
    'HYG': 0.05,
    'LQD': 0.05,
    'TLT': 0.30,
}

start_date = dt.datetime(2007, 7, 1)
prices = mkt_data.get_equity_prices(portfolio_target_weights.keys(),
                                    start_date=start_date)
prices = prices['adj_close'].unstack(level='ticker')

daily_returns = prices / prices.shift(1) - 1
cumulative_returns = daily_returns.cumsum()
dates = daily_returns.index
monthly_dates = utils.closest_business_day(
    daily_returns.resample('M').last().index)

# Continuously rebalanced portfolio
portfolio_daily_returns = pd.DataFrame(index=daily_returns.index,
                                       columns=["daily_return"])
portfolio_daily_returns['daily_return'] = 0
for ticker in portfolio_target_weights.keys():
    portfolio_daily_returns['daily_return'] += daily_returns[ticker] \
        * portfolio_target_weights[ticker]

cumulative_portfolio_returns = pd.DataFrame(index=dates)
cumulative_portfolio_returns['continuous'] = (1 + portfolio_daily_returns).cumprod() - 1

# Properly rebalanced portfolio
portfolio_weights = pd.DataFrame(index=dates,
                                 columns=portfolio_target_weights.keys())
days_since_rebalance = pd.Series(index=dates)
portfolio_weights.iloc[0] = portfolio_target_weights
days_since_rebalance.iloc[0] = 0
for i in range(1, len(dates)):
    date = dates[i]
    days_since_rebalance.iloc[i] = days_since_rebalance.iloc[i-1] + 1
    if days_since_rebalance.loc[date] >= rebalance_frequency:
        days_since_rebalance.loc[date] = 0
        portfolio_weights.loc[date] = portfolio_target_weights
    portfolio_weights.iloc[i] = portfolio_weights.iloc[i-1] \
                              * (1 + daily_returns.loc[dates[i]])
    portfolio_weights.iloc[i] /= portfolio_weights.iloc[i].sum()

# Non-rebalanced portfolio
dynamic_portfolio_weights = portfolio_weights.copy(deep=True)
dynamic_portfolio_returns = (portfolio_weights * daily_returns).sum(axis=1)
dynamic_portfolio_returns = dynamic_portfolio_returns.fillna(value=0)
cumulative_portfolio_returns['rebalance'] = (1 + dynamic_portfolio_returns).cumprod() - 1

# Now keep in mind there are realistically inflows and outflows into this stuff
# And we're really talking about the growth of a cash amount

starting_capital = 25000
inflow_interval = 21
inflow_amount = 1000

inflows = pd.DataFrame(index=monthly_dates)
data = pd.DataFrame(index=dates, columns=['inflows', 'portfolio_value'])
data['inflows'] = 0
data.loc[dates[0], 'inflows'] = starting_capital
data.loc[dates[0], 'portfolio_value'] = starting_capital
for i in range(1, len(dates)):
    date = dates[i]
    if date in inflows.index:
        data.loc[date, 'inflows'] = inflow_amount
    data.loc[date, 'portfolio_return'] = dynamic_portfolio_returns.loc[date]
    data.loc[date, 'portfolio_value'] = \
        data.loc[dates[i-1], 'portfolio_value'] \
        * (1 + data.loc[date, 'portfolio_return']) \
        + data.loc[date, 'inflows']
data['cumulative_inflows'] = data['inflows'].cumsum()

plt.plot(data[['cumulative_inflows', 'portfolio_value']])
plt.legend(["cumulative savings", "portfolio value"])
plt.title("Portfolio value, starting with " + str(starting_capital) + ", saving "
          + str(inflow_amount) + " per month")
