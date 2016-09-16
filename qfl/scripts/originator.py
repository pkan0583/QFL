import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import xmltodict
import simplejson as json

import qfl.core.market_data as md
import qfl.core.portfolio_utils as putils
import qfl.core.database_interface as qfl_data
import qfl.core.market_data as mkt_data
import qfl.core.constants as constants
import qfl.utilities.basic_utilities as utils
import qfl.etl.data_ingest as etl
import qfl.etl.data_interfaces as data_int
from qfl.etl.data_interfaces import YahooApi, QuandlApi, DataScraper, NutmegAdapter
from qfl.core.database_interface import DatabaseInterface as db

reload(qfl_data)
from qfl.etl.data_interfaces import NutmegAdapter

db.initialize()


'''
--------------------------------------------------------------------------------
Nutmeg
--------------------------------------------------------------------------------
'''

# Load file
xml_file = open("data/NH.xml")
na = NutmegAdapter(xml_file=xml_file)

# Account names
na.get_account_names()
account = na.get_account('Savings')

# Fund names
fund_names = na.get_fund_names(account_name='Savings')
fund = na.get_fund('Savings', 'Rainy Day Fund')

# Portfolio data
transactions, cash_flows, security_income, other_adjustments, unique_asset_codes\
    = na.extract_fund_data(account_name='Savings',
                           fund_name='Rainy Day Fund')


'''
--------------------------------------------------------------------------------
Market prices
--------------------------------------------------------------------------------
'''

# Get data from the beginning of the account
start_date = pd.to_datetime(pd.to_datetime(
    na.get_full_dataset()['fromDate']).date())
end_date = pd.to_datetime(pd.to_datetime(
    na.get_full_dataset()['toDate']).date())

# FX rates
gbpusd = QuandlApi.get_currency_prices('GBPUSD', start_date, end_date)
gbpusd = gbpusd[['date', 'last_price']].set_index('date', drop=True)

# Tickers for yahoo and quandl
tickers = [code + '.L' for code in unique_asset_codes]
quandl_tickers = ['LSE/' + code for code in unique_asset_codes]

# Raw quandl data
raw_price_data_ql = QuandlApi.get_data(tickers=quandl_tickers,
                                       start_date=start_date)
price_data = raw_price_data_ql.copy(deep=True).reset_index()

# Pence adjustment
ql_lse = pd.read_csv('data/LSE-datasets-codes.csv', header=None)
quoted_in_pence = ql_lse[ql_lse[1].str.contains('GBX')][0].values.tolist()
ind = price_data.index[price_data['ticker'].isin(quoted_in_pence)]
cols = ['Change', 'High', 'Last Close', 'Low', 'Price']
price_data.loc[ind, cols] /= 100.0

# USD adjustment
quoted_in_usd = ql_lse[ql_lse[1].str.contains('USD')][0].values.tolist()
ind = price_data.index[price_data['ticker'].isin(quoted_in_usd)]
price_data = price_data.join(gbpusd.rename(
    columns={'last_price': 'gbpusd'}), on='date')
price_data['gbpusd'] = price_data['gbpusd']\
    .fillna(method='ffill').fillna(method='bfill')
for col in cols:
    price_data.loc[ind, col] /= price_data.loc[ind, 'gbpusd']

# Revert tickers from Quandl format
price_data['ticker'] = price_data['ticker'].str\
                              .replace('LSE/', '')\
                              .astype(str) + '.L'
price_data = price_data.set_index(['date', 'ticker'], drop=True)
price_data.index.names = ['date', 'asset_id']
price_data = price_data.sort_index()

# Price data to use
price_field = 'Last Close'

# Yield curve
yc_data = DataScraper.retrieve_uk_yield_curve()
uk_cash_rate = yc_data[1].fillna(method='ffill')


'''
--------------------------------------------------------------------------------
Personal Consultant performance analysis
--------------------------------------------------------------------------------
'''

# Next steps here are:
# 1) handle currencies: EG right now USD just a price transform...
# 2) handle asset id for various exchanges

# This is nutmeg: assume everything on London exchange
transactions['asset_id'] = transactions['asset_id'].astype(str) + '.L'

# Summaries
daily_transactions_total = transactions.groupby('date')['market_value'].sum()
daily_cash_flows_total = cash_flows.groupby('date')['market_value'].sum()

# Start date
start_date = pd.to_datetime(cash_flows['date'].min())
end_date = pd.to_datetime(cash_flows['date'].max())
calendar_name = 'UnitedKingdom'

# This is Nutmeg: assume all transactions in GBP
base_currency = 'GBP'
transactions['currency'] = base_currency
cash_flows['currency'] = base_currency
base_asset = 'cash_' + base_currency


positions, account_performance = putils.build_positions_from_transactions(
    end_date=end_date,
    base_currency=base_currency,
    calendar_name=calendar_name,
    cash_flows=cash_flows,
    transactions=transactions,
    security_income=security_income,
    other_adjustments=other_adjustments,
    interest_rates=uk_cash_rate,
    asset_prices=price_data,
    price_field=price_field
)

# Plot of account market value
plt.plot(account_performance['market_value'])
plt.ylabel('account value, GBP')
plt.title('Reconstructed Savings Fund Value (actual end = 155,477)')

# Risk measures
vol_window_days = 21
drawdown_pctile = 0.05
ann_factor = np.sqrt(constants.trading_days_per_year / vol_window_days)
window_returns = account_performance['pnl_pct'].rolling(
    window=vol_window_days, center=False).sum()
window_returns = window_returns[np.isfinite(window_returns)]
account_volatility = window_returns.std()
account_drawdown = window_returns.quantile(drawdown_pctile)

'''
--------------------------------------------------------------------------------
Benchmarking analysis
--------------------------------------------------------------------------------
'''

rebalance_frequency_days = 63

# Diversified global portfolio

global_portfolio_target_weights = {
    'VT': 0.20,
    'ACWI': 0.20,
    'ISF.L': 0.10,
    'EEM': 0.10,
    'HYG': 0.05,
    'EMB': 0.05,
    'LQD': 0.05,
    'IGOV': 0.15,
    'AGG': 0.10
}

prices = YahooApi.retrieve_prices(global_portfolio_target_weights.keys(),
                                  start_date=start_date,
                                  end_date=end_date)
prices.index.names = ['date', 'ticker']
prices = prices['adj_close'].unstack(level='ticker')

cumulative_portfolio_returns, portfolio_returns, portfolio_weights \
    = putils.backtest_equity_portfolio(
        portfolio_target_weights=global_portfolio_target_weights,
        stock_prices=prices,
        rebalance_frequency_days=rebalance_frequency_days
    )

# UK-focused portfolio

uk_portfolio_target_weights = {
    'ISF.L': 0.35,
    'CUKS.L': 0.30,
    'ACWI': 0.35,
    'SLXX.L': 0.0,
    'IGLT.L': 0.0,
    'IGOV': 0.0,
    'AGG': 0.0,
}

prices_uk = YahooApi.retrieve_prices(uk_portfolio_target_weights.keys(),
                                     start_date=start_date,
                                     end_date=end_date)
prices_uk.index.names = ['date', 'ticker']
prices_uk = prices_uk['adj_close'].unstack(level='ticker')

cumulative_portfolio_returns_uk, portfolio_returns_uk, portfolio_weights_uk \
    = putils.backtest_equity_portfolio(
        portfolio_target_weights=uk_portfolio_target_weights,
        stock_prices=prices_uk,
        rebalance_frequency_days=rebalance_frequency_days
    )

uk_bm_window_returns = portfolio_returns_uk.rolling(
    window=vol_window_days, center=False).sum()
uk_bm_window_returns = uk_bm_window_returns[np.isfinite(uk_bm_window_returns)]
uk_bm_volatility = uk_bm_window_returns.std() * ann_factor
uk_bm_drawdown = uk_bm_window_returns.quantile(drawdown_pctile)



# Plot of account percent pnl versus benchmark
fig, ax1 = plt.subplots()
plt.title('Cumulative Investment Returns')
ax1.plot(account_performance['cum_pnl_pct'] * 100.0)
ax1.plot(cumulative_portfolio_returns_uk['rebalance'] * 100.0)
ax1.set_ylabel('cumulative compound returns, %')
ax1.yticks = mtick.FormatStrFormatter('%.0f%%')
plt.legend(['wealth manager track', 'similar-risk benchmark'])


# Now keep in mind there are realistically inflows and outflows into this stuff
# And we're really talking about the growth of a cash amount

# starting_capital = 25000
# inflow_interval = 21
# inflow_amount = 1000
#
# inflows = pd.DataFrame(index=monthly_dates)
# data = pd.DataFrame(index=dates, columns=['inflows', 'portfolio_value'])
# data['inflows'] = 0
# data.loc[dates[0], 'inflows'] = starting_capital
# data.loc[dates[0], 'portfolio_value'] = starting_capital
# for i in range(1, len(dates)):
#     date = dates[i]
#     if date in inflows.index:
#         data.loc[date, 'inflows'] = inflow_amount
#     data.loc[date, 'portfolio_return'] = dynamic_portfolio_returns.loc[date]
#     data.loc[date, 'portfolio_value'] = \
#         data.loc[dates[i-1], 'portfolio_value'] \
#         * (1 + data.loc[date, 'portfolio_return']) \
#         + data.loc[date, 'inflows']
# data['cumulative_inflows'] = data['inflows'].cumsum()
#
# plt.plot(data[['cumulative_inflows', 'portfolio_value']])
# plt.legend(["cumulative savings", "portfolio value"])
# plt.title("Portfolio value, starting with " + str(starting_capital) + ", saving "
#           + str(inflow_amount) + " per month")
