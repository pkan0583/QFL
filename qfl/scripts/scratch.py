
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
import qfl.core.calcs as calcs
import qfl.core.market_data as md
import qfl.macro.macro_models as macro
import qfl.etl.data_ingest as etl
import qfl.core.calcs as lib
import qfl.utilities.basic_utilities as utils
import qfl.core.constants as constants
from scipy import interpolate
import matplotlib.ticker as mtick
from matplotlib import cm
import logging
from pandas.tseries.offsets import BDay
from scipy import interpolate

reload(calcs)
reload(utils)
reload(qfl_data)
reload(data_int)
reload(md)
reload(etl)

from qfl.core.data_interfaces import QuandlApi, YahooApi, IBApi, FigiApi, DataScraper
from qfl.core.database_interface import DatabaseInterface as db, DatabaseUtilities as dbutils
from qfl.utilities.chart_utilities import format_plot
from qfl.utilities.nlp import DocumentsAnalyzer as da

from collections import defaultdict
from gensim import corpora, models, similarities
import odo

db.initialize()

# Mapping
etl.process_optionworks_mappings()


'''
--------------------------------------------------------------------------------
Nutmeg export
--------------------------------------------------------------------------------
'''

import xmltodict
import simplejson as json

xml_file = open("data/AM.xml")
am_dict = xmltodict.parse(xml_file)

xml_file = open("data/NH.xml")
nh_dict = xmltodict.parse(xml_file)

nh_data = nh_dict['userSummaryAdmins']['userSummaryAdmin']
nh_account_data = nh_data['accounts']['account']
test_json = json.dumps(nh_data)
test_json_extract = json.loads(test_json)

# Example funds with relatively large balances (main funds)
nh_savings_fund_general = nh_account_data[1]['funds']['fund'][6]
nh_pension_fund_general = nh_account_data[0]['funds']['fund'][2]

# Contributions
nh_contributions = nh_savings_fund_general['contributionActivity']['postingSummary']

# Mapping contributions data into tabular format
nh_c_df = pd.DataFrame(index=range(0, len(nh_contributions)),
                       columns=nh_contributions[0].keys())
for i in range(0, len(nh_contributions)):
    nh_c_df.loc[i] = nh_contributions[i].values()

# Some formatting issues
nh_c_df['value'] = pd.to_numeric(nh_c_df['value'])
nh_c_df['postedDate'] = pd.to_datetime(pd.to_datetime(
    nh_c_df['postedDate']).dt.date)
nh_c_df = nh_c_df.sort_values('postedDate')

# Transactions
nh_transactions = nh_savings_fund_general['stockSummaries']['stockSummary']

# Transaction data columns
columns = ['sequence',
           'postedDate',
           'transactionType',
           'transactionRef',
           'assetCode',
           'units',
           'value']

# Mapping the transaction data into tabular format
nh_t_df = pd.DataFrame(columns=columns)
counter = 0
for i in range(0, len(nh_transactions)):
    posted_date = nh_transactions[i]['postedDate']
    entries = nh_transactions[i]['entries']
    for j in range(0, len(entries)):
        if 'transactionType' in entries[j]:
            nh_t_df.loc[counter] = entries[j].values()
            counter += 1

# Some formatting issues
nh_t_df['value'] = pd.to_numeric(nh_t_df['value'])
nh_t_df['units'] = pd.to_numeric(nh_t_df['units'])
nh_t_df['postedDate'] = pd.to_datetime(pd.to_datetime(
    nh_t_df['postedDate']).dt.date)
nh_t_df = nh_t_df.sort('postedDate')

# Security code universe
unique_asset_codes = np.unique(nh_t_df['assetCode'])

# I'm not exactly sure how this is different... non-stock trades? AH DIVS ETC
nh_investments = nh_savings_fund_general['investmentActivity']['postingSummary']
nh_i_df = pd.DataFrame(index=range(0, len(nh_investments)),
                       columns=nh_investments[0].keys() + ['transactionText'])
for i in range(0, len(nh_investments)):
    nh_i_df.loc[i] = nh_investments[i]
nh_i_df['value'] = pd.to_numeric(nh_i_df['value'])
nh_i_df['units'] = pd.to_numeric(nh_i_df['units'])
nh_i_df['postedDate'] = pd.to_datetime(
    pd.to_datetime(nh_i_df['postedDate']).dt.date)
nh_i_df = nh_i_df.sort_values('postedDate')

# Identify dividends
nh_i_df['transactionText'] = nh_i_df['transactionText'].fillna(value='')
nh_divs = nh_i_df[nh_i_df['transactionText'].str.contains('Dividend')]
nh_adjs = nh_i_df[nh_i_df['transactionText'].str.contains('Adjustment')]

# Export
nh_c_df.to_excel("data/nh_contributions.xlsx")
nh_t_df.to_excel("data/nh_transactions.xlsx")

'''
--------------------------------------------------------------------------------
Market prices
--------------------------------------------------------------------------------
'''

# Get data from the beginning of the account
start_date = nh_data['fromDate']

# Tickers for yahoo and quandl
tickers = [code + '.L' for code in unique_asset_codes]
quandl_tickers = ['LSE/' + code for code in unique_asset_codes]

# Raw quandl data
raw_price_data_ql = QuandlApi.get_data(tickers=quandl_tickers,
                                       start_date=start_date)
price_data_ql = raw_price_data_ql.copy(deep=True).reset_index()

# Adjust stuff that is quoted in Pence
ql_lse = pd.read_csv('data/LSE-datasets-codes.csv', header=None)
quoted_in_pence = ql_lse[ql_lse[1].str.contains('GBX')][0].values.tolist()
quoted_in_usd = ql_lse[ql_lse[1].str.contains('USD')].values.tolist()
ind = price_data_ql.index[price_data_ql['ticker'].isin(quoted_in_pence)]
price_data_ql.loc[ind, ['Change', 'High', 'Last Close', 'Low', 'Price']] /= 100.0

# Revert tickers
price_data_ql['ticker'] = price_data_ql['ticker'].str\
                              .replace('LSE/', '')\
                              .astype(str) + '.L'
price_data_ql = price_data_ql.set_index(['date', 'ticker'], drop=True)
price_data_ql.index.names = ['date', 'asset_id']
price_data_ql = price_data_ql.sort_index()

# Try to get some data
price_field_yf = 'Adj Close'
raw_price_data_yf = pdata.get_data_yahoo(tickers, start=start_date)
price_data_yf = raw_price_data_yf.to_frame()[price_field_yf]
price_data_yf.index.names = ['date', 'asset_id']

price_tickers = np.unique(price_data_yf.index.get_level_values('asset_id'))
price_tickers = [str(ticker) for ticker in price_tickers]

# Missing
missing_tickers = list(set(tickers) - set(price_tickers))

# Price data to use
price_data = price_data_ql
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
# 1) "market value" --> needs to reflect market prices not transactions
# 2) need to adjust cash to reflect transactions
# 3) handle currencies
# 4) handle asset id for various exchanges

reload(utils)

cash_flows = nh_c_df.sort_values('postedDate')
transactions = nh_t_df.sort_values('postedDate')

# Standardize columns
col_map = {'postedDate': 'date',
           'units': 'quantity',
           'value': 'market_value',
           'assetCode': 'asset_id'}
cash_flows = cash_flows.rename(columns=col_map)
transactions = transactions.rename(columns=col_map)
security_income = nh_divs.rename(columns=col_map)
other_adjustments = nh_adjs.rename(columns=col_map)

# This is nutmeg: assume everything on London exchange
transactions['asset_id'] = transactions['asset_id'].astype(str) + '.L'

# Standardize date formats
cash_flows['date'] = pd.to_datetime(cash_flows['date'].dt.date)
transactions['date'] = pd.to_datetime(transactions['date'].dt.date)

# Change sells to negative quantity
ind = transactions.index[transactions['transactionType'] == 'SLD']
transactions.loc[ind, ['quantity', 'market_value']] *= -1.0

# Start date
start_date = pd.to_datetime(cash_flows['date'].min())
end_date = pd.to_datetime(cash_flows['date'].max())
calendar_name = 'UnitedKingdom'

# This is Nutmeg: assume all transactions in GBP
base_currency = 'GBP'
transactions['currency'] = base_currency
cash_flows['currency'] = base_currency
base_asset = 'cash_' + base_currency

# Columns
cols = ['date', 'quantity', 'market_value']

# Dates
dates = utils.DateUtils.get_business_date_range(start_date,
                                                end_date,
                                                calendar_name)
start_date = dates[0]

# Initial positions
initial_positions = pd.DataFrame(index=[pd.Series(base_asset)],
                                 columns=cols)
initial_positions.index.names = ['asset_id']
initial_positions.loc[base_asset, ['quantity', 'market_value']] \
    = cash_flows[cash_flows['date'] == start_date]['market_value'].values[0]
initial_positions.loc[base_asset, 'date'] = start_date

# Iterate over dates
positions_dict = dict()
positions_dict[0] = initial_positions
for t in range(1, len(dates)):

    # Starting point: carry over positions
    positions_dict[t] = positions_dict[t-1].copy(deep=True)
    positions_dict[t]['date'] = dates[t]

    # Cash balances accrue interest
    elapsed_days = (dates[t] - dates[t-1]).days
    daily_rate = uk_cash_rate[(uk_cash_rate.index >= dates[t - 1])
                            & (uk_cash_rate.index < dates[t])].mean()

    # Carry over prior value
    if np.isnan(daily_rate):
        daily_rate = prev_daily_rate
    prev_daily_rate = daily_rate

    positions_dict[t].loc[base_asset, ['quantity', 'market_value']] \
        *= (1 + daily_rate) * elapsed_days / 365.0

    # New cash flows
    # TODO: what if the cash flow is in a different currency
    cf = cash_flows[(cash_flows['date'] <= dates[t])
                  & (cash_flows['date'] > dates[t - 1])]
    positions_dict[t].loc[base_asset, ['quantity', 'market_value']]\
        += cf['market_value'].sum()

    # Security income
    # TODO: what if the cash flow is in a different currency
    si = security_income[(security_income['date'] <= dates[t])
                       & (security_income['date'] > dates[t - 1])]
    positions_dict[t].loc[base_asset, ['quantity', 'market_value']]\
        += si['market_value'].sum()

    # Adjustments
    # TODO: what if the cash flow is in a different currency
    oa = other_adjustments[(other_adjustments['date'] <= dates[t])
                         & (other_adjustments['date'] > dates[t - 1])]
    positions_dict[t].loc[base_asset, ['quantity', 'market_value']]\
        += oa['market_value'].sum()

    # New transactions
    tt = transactions[(transactions['date'] <= dates[t])
                    & (transactions['date'] > dates[t - 1])]
    if len(tt) > 0:

        # Debit cash account for transactions
        positions_dict[t].loc[base_asset, ['quantity', 'market_value']]\
            -= tt['market_value'].sum()

        # Changes in existing positions
        ind = tt.index[tt['asset_id'].isin(positions_dict[t].index)]
        if len(ind) > 0:
            asset_codes = tt.loc[ind, 'asset_id']
            positions_dict[t].loc[asset_codes, ['quantity', 'market_value']] \
                += tt.loc[ind, ['quantity', 'market_value']].values
        # New positions
        new_ind = tt.index[~tt['asset_id'].isin(positions_dict[t].index)]
        if len(new_ind) > 0:
            asset_codes = tt.loc[new_ind, 'asset_id']
            new_df = pd.DataFrame(index=asset_codes,
                                  columns=cols,
                                  data=tt.loc[new_ind, cols]
                                  .values)
            positions_dict[t] = positions_dict[t].append(new_df)

    # Mark the book
    price_data_date = price_data[
        price_data.index.get_level_values('date') == dates[t]].reset_index()
    price_data_date.index = price_data_date['asset_id']
    ind = positions_dict[t].index[
        positions_dict[t].index.isin(price_data_date.index)]
    positions_dict[t].loc[ind, 'market_value'] = \
        positions_dict[t].loc[ind, 'quantity'] \
        * price_data_date.loc[ind, price_field]

positions = pd.concat(positions_dict).reset_index()
positions.index = [positions['date'], positions['asset_id']]
positions = positions.rename(columns={'level_0': 'date_index'})

# Join positions data to price for visual inspection
positions = positions.join(price_data['Price'])
positions[positions.index.get_level_values('asset_id') == base_asset] = 1.0
del positions['asset_id']
del positions['date']

account_performance = pd.DataFrame(positions['market_value'].groupby(level='date').sum())
account_performance['daily_flows'] = cash_flows.groupby('date')['market_value'].sum()
account_performance['daily_flows'] = account_performance['daily_flows'] .fillna(value=0)
account_performance['pnl'] = account_performance['market_value'].diff(1)\
                             - account_performance['daily_flows']
account_performance['pnl_pct'] = account_performance['pnl'] \
                                 / account_performance['market_value'].shift(1)

# plt.plot((1 + account_performance['pnl_pct']).cumprod()-1)
plt.plot(account_performance['market_value'])
plt.ylabel('account value, GBP')
plt.title('Reconstructed Savings Fund Value (actual end = 155,477)')

'''
--------------------------------------------------------------------------------
OptionWorks data
--------------------------------------------------------------------------------
'''

# data = md.get_optionworks_staging_ivm(codes=['CME_ES_EW4_'])

futures_series = 'C'
maturity_type = 'futures_contract'
start_date = dt.datetime(2009, 1, 1)

etl.process_optionworks_data_ivs(series_id=futures_series,
                                 start_date=start_date,
                                 maturity_type='constant_maturity')

'''
--------------------------------------------------------------------------------
Positioning analysis
--------------------------------------------------------------------------------
'''

futures_series = 'VX'
start_date = dt.datetime(2010, 1, 1)
vix_tenor_days = 21
vol_lookahead_days = 21
vol_trailing_days = 63

cftc_data = md.get_cftc_positioning_by_series(futures_series, start_date)
cftc_data.index = cftc_data['date']

lev_net = cftc_data["Lev_Money_Positions_Long_All"] \
          - cftc_data['Lev_Money_Positions_Short_All']
lev_net.name = 'lev_net'

vix_futures_px = md.get_constant_maturity_futures_prices_by_series(
    futures_series='VX', start_date=start_date)['price']\
    .unstack('days_to_maturity')

# Rolling returns
price_field = 'settle_price'
level_change = True
days_to_zero_around_roll = 1
futures_returns = md.get_rolling_futures_returns_by_series(
    futures_series='VX',
    start_date=start_date,
    level_change=True
)

data = pd.DataFrame(vix_futures_px[vix_tenor_days]).reset_index()
data.index = data['date']
del data['date']

data = data.join(lev_net).join(futures_returns)
data = data.reset_index().drop_duplicates('date')
data.index = data['date']
data = data.rename(columns={vix_tenor_days: price_field})

data['lev_net'] = data['lev_net'].fillna(method='ffill')
data['lev_net_z'] = (data['lev_net'] - data['lev_net'].mean()) \
                    / data['lev_net'].std()

data['fut_pnl'] = data['VX1'].shift(-vol_lookahead_days)\
                             .rolling(window=vol_lookahead_days)\
                             .sum()
data['cum_fut_pnl'] = data['VX1'].cumsum()
data['vol'] = data['VX1'].rolling(window=vol_trailing_days,
                                  center=False).std()
data['fut_vol'] = data['vol'].shift(-vol_trailing_days)

t1 = pd.ols(y=data['fut_pnl'], x=data[[price_field, 'lev_net_z']])
t2 = pd.ols(y=data['fut_vol'], x=data[[price_field, 'vol', 'lev_net_z']])


plot_cols = ['VX1', 'VX2', 'VX3', 'VX4', 'VX5', 'VX6']
plt.plot(futures_returns[plot_cols].cumsum())
plt.legend(plot_cols)

from statsmodels import regression as reg
from statsmodels import tools as smtools
exog = data[[vix_tenor_days, 'lev_net_z']].values
exog = smtools.tools.add_constant(exog)
t3 = reg.linear_model.GLSAR(endog=data['fut_pnl'].values,
                            exog=exog,
                            missing='drop',
                            rho=5,
                            hasconst=True)

# Neuberger overlapping estimator
# for t in range(0, len(data)):



'''
--------------------------------------------------------------------------------
NLP for matching futures descriptions
--------------------------------------------------------------------------------
'''


s = 'select distinct id, series, description from futures_series'
qfl_1 = db.read_sql(s)

s = 'select distinct "Market_and_Exchange_Names", "CFTC_Contract_Market_Code",' \
    ' "CFTC_Market_Code"' \
    'from staging_cftc_positioning'
cftc_1 = db.read_sql(s)

s = 'select distinct "Market_and_Exchange_Names", "CFTC_Contract_Market_Code",' \
    ' "CFTC_Market_Code"' \
    'from staging_cftc_positioning_financial'
cftc_2 = db.read_sql(s)

cftc = cftc_1.append(cftc_2)
# documents = cftc['Market_and_Exchange_Names'].values.tolist()
documents = qfl_1['description'].values.tolist()

# Hyphens are really screwing this all up, let's kill them
for i in range(0, len(documents)):
    documents[i] = documents[i].replace('-', ' ')

common_words = set('for a of the and to in - '.split())

texts = da.remove_common_words(documents, common_words)
texts = da.remove_solitary_words(texts)
dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=20)



matches = 3
cftc.index = range(0, len(cftc))
match_df = cftc.copy(deep=True)


# convert the query to LSI space
for i in range(544, len(cftc)):

    df = pd.DataFrame(index=range(0, len(documents)),
                      columns=['overlap', 'description', 'series', 'id'])
    series_description = cftc.iloc[i]['Market_and_Exchange_Names']
    series_description = series_description.replace('-', ' ')
    for k in range(0, 5):
        series_description = series_description.replace("  ", " ")
    series_des_analyze = da.remove_common_words(
        series_description.lower().split(), common_words=common_words)
    series_des_analyze = [s[0] for s in series_des_analyze
                          if len(s) > 0]
    print(series_description)

    counter = 0
    for text in texts:
        _int = set(series_des_analyze).intersection(set(text))
        df.loc[counter, 'overlap'] = len(_int)
        df.loc[counter, 'description'] = documents[counter]
        df.loc[counter, 'series'] = qfl_1.loc[counter, 'series']
        df.loc[counter, 'id'] = qfl_1.loc[counter, 'id']
        counter += 1

    vec_bow = dictionary.doc2bow(series_description.lower().split())
    vec_lsi = lsi[vec_bow]

    # perform a similarity query against the corpus
    index = similarities.MatrixSimilarity(lsi[corpus])
    sims = index[vec_lsi]

    sims_df = pd.DataFrame(sims, columns=['similarity'])
    sims_df['name'] = documents
    sims_df.sort_values('similarity', ascending=False).head()
    df['lda_similarity'] = sims_df['similarity']

    df = df.sort_values(['overlap', 'lda_similarity'], ascending=False)

    for m in range(0, matches):
        match_df.loc[i, 'match_' + str(m)] = df.iloc[m]['description']
        match_df.loc[i, 'series_' + str(m)] = df.iloc[m]['series']

        match_df.loc[i, 'overlap_' + str(m)] = df.iloc[m]['overlap']
        match_df.loc[i, 'lda_similarity' + str(m)] = df.iloc[m]['lda_similarity']

match_df.to_excel('data/cftc_qfl_code_mapping.xlsx')

'''
--------------------------------------------------------------------------------
GENSIM tutorial
--------------------------------------------------------------------------------
'''

from gensim import corpora, models, similarities

documents = ["Human machine interface for lab abc computer applications",
          "A survey of user opinion of computer system response time",
          "The EPS user interface management system",
          "System and human system engineering testing of EPS",
          "Relation of user perceived response time to error measurement",
          "The generation of random binary unordered trees",
          "The intersection graph of paths in trees",
          "Graph minors IV Widths of trees and well quasi ordering",
          "Graph minors A survey"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
      for document in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
 for token in text:
     frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
      for text in texts]

from pprint import pprint  # pretty-printer
pprint(texts)

# store the dictionary, for future reference
dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict')
pprint(dictionary)
pprint(dictionary.token2id)

# the word "interaction" does not appear in the dictionary and is ignored
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

# store to disk, for later use
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)

class MyCorpus(object):
     def __iter__(self):
         for line in open('mycorpus.txt'):
             # assume there's one document per line, tokens separated by whitespace
             yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
print(corpus_memory_friendly)

dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
corpus = corpora.MmCorpus('/tmp/deerwester.mm')
print(corpus)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

# convert the query to LSI space
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]
print(vec_lsi)

# transform corpus to LSI space and index it
index = similarities.MatrixSimilarity(lsi[corpus])

# Saving and loading index
index.save('/tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')

# perform a similarity query against the corpus
sims = index[vec_lsi]
pprint(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])
pprint(sims)

'''
--------------------------------------------------------------------------------
Rolling PCA?
--------------------------------------------------------------------------------
'''

etf_tickers, etf_exchange_codes = md.get_etf_universe()

prices = md.get_equity_prices(tickers=etf_tickers,
                              start_date=md.history_default_start_date)
prices = prices['adj_close'].unstack('ticker')

# Every month we run a PCA on the last year's overlapping weekly returns


'''
--------------------------------------------------------------------------------
Get latest VIX termstructure
--------------------------------------------------------------------------------
'''

date = dt.date.today()

prices = md.get_constant_maturity_futures_prices_by_series(futures_series='VX')
prices = md.get_futures_prices_by_series(futures_series='VX')

ind = prices.index[prices.index.get_level_values('date') >= pd.to_datetime(date)]
prices.loc[ind, ['days_to_maturity', 'seasonality_adj_price']]\
    .sort_values('days_to_maturity')


ts = prices.loc[ind, ['days_to_maturity', 'seasonality_adj_price']]\
    .sort_values('days_to_maturity')

fig, ax = plt.subplots(figsize=[10, 6])
plt.scatter(ts['days_to_maturity'], ts['seasonality_adj_price'], color='r')
format_plot(ax, 'red', 'black')

'''
--------------------------------------------------------------------------------
Historical return distributions
--------------------------------------------------------------------------------
'''

ticker = 'TLT'
price_field = 'last_price'
start_date = dt.datetime(2000, 1, 1)
return_window_days = 126

prices = md.get_equity_prices(ticker,
                              price_field=price_field,
                              start_date=start_date)
prices = prices[price_field]
returns = prices / prices.shift(return_window_days) - 1

returns = returns[np.isfinite(returns)]
plt.hist(returns, bins=20)

# Neutralize the rates trend over the period
demeaned_returns = returns - returns.mean(axis=0)
plt.hist(demeaned_returns, bins=20)


'''
--------------------------------------------------------------------------------
INTERACTIVE BROKERS API
--------------------------------------------------------------------------------
'''

ib_api = IBApi()
ib_api.initialize()

contracts_df, contracts = ib_api.retrieve_contracts('SPY', 'USD', 'OPT')

prices_df, prices = ib_api.retrieve_prices(
    underlying_ticker='SPY',
    maturity_date='20161216',
    strike_price=200,
    option_type='P',
    security_type='OPT',
    exchange='SMART',
    currency='USD',
    multiplier=100,
    subscribe=True
)

prices_df, prices = ib_api.retrieve_prices(
    underlying_ticker='SPY',
    security_type='OPT',
    exchange='SMART',
    currency='USD',
    strike_price=200,
    maturity_date='20161216',
    option_type='P',
    multiplier=100
)

test_contract = ib_api.create_contract(
    symbol='SPY',
    sec_type='STK',
    exch='SMART',
    prim_exch='SMART',
    curr='USD'
)

duration_str = "10 D"
bar_size = '1 hour'

historical_data, raw_historical_data = \
    ib_api.retrieve_historical_prices(
        ticker='SPY',
        multiplier=100,
        strike_price=200,
        maturity_date='20161216',
        option_type='P',
        currency='USD',
        security_type='OPT',
        exchange='SMART',
        duration_str="1 M",
        bar_size="30 mins"
)

historical_data = ib_api.process_historial_prices(ib_api.historical_prices)

# etl.add_equities_from_index(ticker='NDX', country='US', _db=db)

date = dt.datetime.today()
source_series = 'EUREX_FVS'
futures_series = 'FVS'
dataset = 'CHRIS'
_db = db
contract_range = np.arange(0, 8)
start_date = dt.datetime(2012, 1,1 )

etl.ingest_historical_futures_prices(
    dataset=dataset,
    futures_series=futures_series,
    start_date=dt.datetime(2012, 1, 1)
)

etl.ingest_historical_generic_futures_prices(
    dataset=dataset,
    contract_range=contract_range,
    start_date=start_date,
    futures_series=futures_series,
    source_series=source_series
)


'''
--------------------------------------------------------------------------------
MACRO
--------------------------------------------------------------------------------
'''

reload(macro)

start_date = dt.datetime(1980, 1, 1)
ffill_limit = 3
diff_window = 1
diff_threshold = 0.875

macro_data, settings, raw_macro_data = macro.load_fundamental_data()

macro_data, macro_data_1d, macro_data_3d, macro_data_6d, macro_data_ma = \
    macro.process_factor_model_data(raw_macro_data=raw_macro_data,
                                    macro_data=macro_data,
                                    settings=settings)

# AR1
ar1_coefs = macro.compute_ar_coefs(macro_data=macro_data,
                                   macro_data_1d=macro_data_1d,
                                   macro_data_3d=macro_data_3d,
                                   macro_data_6d=macro_data_6d,
                                   macro_data_ma=macro_data_ma,
                                   settings=settings)

macro_data_stationary = macro.process_data_by_ar(macro_data=macro_data,
                                                 ar1_coefs=ar1_coefs,
                                                 ar1_threshold=diff_threshold,
                                                 diff_window=diff_window)

# Z-scores
macro_data_z = ((macro_data_stationary - macro_data_stationary.mean())
    / macro_data_stationary.std()).fillna(method='ffill', limit=ffill_limit)
components, pca_factors, pca_obj = macro.compute_factors(
    macro_data_z=macro_data_z,
    settings=settings)
components, pca_factors = macro.process_factors(
    components=components,
    pca_factors=pca_factors,
    settings=settings)
# MA Z-scores
macro_data_maz = ((macro_data_ma - macro_data_ma.mean()) / macro_data_ma.std()
                  ).fillna(method='ffill', limit=ffill_limit)
components_ma, pca_factors_ma, pca_obj_ma = macro.compute_factors(
    macro_data_z=macro_data_maz, settings=settings)
components_ma, pca_factors_ma = macro.process_factors(
    components=components_ma, pca_factors=pca_factors_ma, settings=settings)

plot_start_date = dt.datetime(1980, 1, 1)
plt.figure()
plt.plot(pca_factors[pca_factors.index >= plot_start_date][0])
plt.plot(pca_factors[0].rolling(window=3, center=False).mean())


volatility_indicator='Volatility252'
macro_indicator = pca_factors[0]

capped_volatility, capped_volatility_lead, residual_volatility = \
    macro.prepare_volatility_data(ticker='^GSPC',
                                  start_date=start_date,
                                  exclude87=True)

f, f_res, out_grid, npd = macro.prepare_nonparametric_analysis(
    capped_volatility_lead=capped_volatility_lead,
    capped_volatility=capped_volatility,
    residual_volatility=residual_volatility,
    macro_indicator=macro_indicator,
    start_date=start_date
)


##############################################################################
# Plotting
##############################################################################

plot_start_date = dt.datetime(2010, 1, 1)
plot_ind = npd.index[npd.index >= plot_start_date]
plot_data = npd.loc[plot_ind, ['ZF1M', 'ZLF1']]
text_color_scheme = 'red'
background_color = 'black'

f1, ax1 = plt.subplots(figsize=[10, 6])
plt.plot(plot_data)
ax1.set_title('US Macro Model',
              fontsize=14,
              fontweight='bold')
ax1.title.set_color(text_color_scheme)
ax1.spines['bottom'].set_color(text_color_scheme)
ax1.spines['top'].set_color(text_color_scheme)
ax1.xaxis.label.set_color(text_color_scheme)
ax1.yaxis.label.set_color(text_color_scheme)
ax1.set_ylabel('Z-Score')
ax1.tick_params(axis='x', colors=text_color_scheme)
ax1.tick_params(axis='y', colors=text_color_scheme)
# ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
leg = ax1.legend(['Level', 'Moving Average'], loc=3)
ax1.set_axis_bgcolor(background_color)
leg.get_frame().set_facecolor('red')
plt.savefig('figures/macro.png',
            facecolor='k',
            edgecolor='k',
            transparent=True)

##############################################################################
# Plotting
##############################################################################

from mpl_toolkits.mplot3d import Axes3D

currentDate = npd.index[len(npd.index)-1]
currentLevel = npd['ZF1M'][currentDate]
currentChange = npd['ZDF1'][currentDate]

# Transparent colormap
theCM = cm.get_cmap()
theCM._init()
alphas = np.abs(np.linspace(-2.0, 2.0, theCM.N))
theCM._lut[:-3,-1] = alphas

# Figure 1: nonparametric
fig = plt.figure(figsize=(16,8))
ax = fig.gca(projection='3d')
scat = ax.scatter(currentLevel,
                  currentChange,
                  f(currentLevel, currentChange) * 100 + 2,
                  c='r',
                  s=200,
                  zorder=10)
surf = ax.plot_surface(outGrid[0], outGrid[1], predGrid * 100.0,
                       rstride=1, cstride=1, cmap='coolwarm',
                       linewidth=0, antialiased=False, zorder=0)
ax.set_xlabel('F1')
ax.set_ylabel('Change in F1')
ax.set_zlabel('S&P realized volatility, next 1y')
ax.set_title('Subsequent 1y S&P realized volatility,'
             'versus Z-scores of level and change of US macro index')
ax.set_xlim([changeGridMin, changeGridMax])
ax.set_ylim([levelGridMin, levelGridMax])
# ax.set_zlim([10, 60])
fmt = '%.0f%%'
zticks = mtick.FormatStrFormatter(fmt)
ax.zaxis.set_major_formatter(zticks)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_axis_bgcolor('w')
# ax.axis([-3, 3, -3, 3, 10, 40])
ax.view_init(30, 30)
plt.savefig('figures/macro_volatility.png',
            facecolor='w',
            edgecolor='w',
            transparent=True)


print('done!')


# Prediction grid
prediction_lag = 6
macro_pred_grid = pd.DataFrame(index=settings.data_fields,
                               columns=settings.data_fields)
for f1 in settings.data_fields:
    print('regressions for ' + f1)
    for f2 in settings.data_fields:
        tmp = pd.ols(y=macro_data_z[f1],
                     x=macro_data_z[f2].shift(prediction_lag))
        macro_pred_grid.loc[f1, f2] = tmp.beta[0]


#############################################################################
# AWS
#############################################################################




'''
--------------------------------------------------------------------------------
VIX versus skew
--------------------------------------------------------------------------------
'''
start_date = dt.datetime(1990, 1, 1)
skew = QuandlApi.get('CBOE/SKEW', start_date)
vix = md.get_generic_index_prices('VIX', start_date)['last_price']
vix.index = vix.index.get_level_values('date')
data = pd.DataFrame(vix)
data['SKEW'] = skew
data = data.rename(columns={'last_price': 'VIX'})


plt.scatter(data['VIX'][data.index >= dt.datetime(2010, 1, 1)],
            data['SKEW'][data.index >= dt.datetime(2010, 1, 1)],
            color='b')
plt.scatter(data['VIX'].values[-1],
            data['SKEW'].values[-1],
            color='r')
plt.ylabel('SKEW Index')
plt.xlabel('VIX Index')
plt.title('VIX versus SKEW, 2010-2016')
plt.legend(['historical', 'live'])

data = QuandlApi.get('CBOE/SKEW', etl.default_start_date).reset_index()
s = "select id from generic_indices where ticker = 'SKEW'"
id = db.read_sql(s).iloc[0].values[0]
data['id'] = id
data = data.rename(columns={'SKEW': 'last_price'})
db.execute_db_save(df=data,
                   table=table,
                   extra_careful=False,
                   time_series=True)

'''
--------------------------------------------------------------------------------
Barchart
--------------------------------------------------------------------------------
'''

import barchart
barchart.API_KEY = '5c45079e0956acbcf33925204ee4846a'

import urllib2
import simplejson as json

barchart_api = "getQuote"
return_format = "json"
req_url = "http://marketdata.websol.barchart.com/getQuote.json?key=5c45079e0956acbcf33925204ee4846a&symbols=VIQ16"


response = urllib2.urlopen(req_url)
results_dict = json.loads(response.read())
if 'results' in results_dict:
    results_list = results_dict['results']
    df = pd.DataFrame(results_list)
    df.index = [df['symbol'], df['tradeTimestamp']]
    del df['symbol']
    del df['tradeTimestamp']

import requests_cache
session = requests_cache.CachedSession(cache_name='cache',
    backend='sqlite', expire_after=dt.timedelta(days=1))

# getQuote with ONE symbol
# ========================
symbol = "^EURUSD"
quote = barchart.getQuote(symbol,
                          apikey=barchart.API_KEY,
                          session=session)
print(quote) # quote is a dict



'''
--------------------------------------------------------------------------------
PYMC
--------------------------------------------------------------------------------
'''


lib.plot_invgamma(a=12, b=1)

# Bayesian normal regression example
# NOTE: the linear regression model we're trying to solve for is
# given by:
# y = b0 + b1(x) + error
# where b0 is the intercept term, b1 is the slope, and error is
# the error

float_df = pd.DataFrame()
float_df['weight'] = np.random.normal(0, 1, 100)
float_df['mpg'] = 0.5 * float_df['weight'] + np.random.normal(0, 0.5, 100)

# model the intercept/slope terms of our model as
# normal random variables with comically large variances
b0 = pymc.Normal("b0", 0, 0.0003)
b1 = pymc.Normal("b1", 0, 0.0003)

# model our error term as a uniform random variable
err = pymc.Uniform("err", 0, 500)

# "model" the observed x values as a normal random variable
# in reality, because x is observed, it doesn't actually matter
# how we choose to model x -- PyMC isn't going to change x's values
x_weight = pymc.Normal("weight", 0, 1, value=np.array(float_df["weight"]), observed=True)

# this is the heart of our model: given our b0, b1 and our x observations, we want
# to predict y
@pymc.deterministic
def pred(b0=b0, b1=b1, x=x_weight):
    return b0 + b1*x

# "model" the observed y values: again, I reiterate that PyMC treats y as
# evidence -- as fixed; it's going to use this as evidence in updating our belief
# about the "unobserved" parameters (b0, b1, and err), which are the
# things we're interested in inferring after all
y = pymc.Normal("y", pred, err, value=np.array(float_df["mpg"]), observed=True)

# put everything we've modeled into a PyMC model
model = pymc.Model([pred, b0, b1, y, err, x_weight])

mc = pymc.MCMC(model)
mc.sample(10000)
print np.mean(mc.trace('b1')[:])
plt.hist(mc.trace('b1')[:], bins=50)

print(__doc__)

# Authors: Alexandre Gramfort
#          Denis A. Engemann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

###############################################################################
# Create the data

n_samples, n_features, rank = 1000, 50, 10
sigma = 1.
rng = np.random.RandomState(42)
U, _, _ = linalg.svd(rng.randn(n_features, n_features))
X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)

# Adding homoscedastic noise
X_homo = X + sigma * rng.randn(n_samples, n_features)

# Adding heteroscedastic noise
sigmas = sigma * rng.rand(n_features) + sigma / 2.
X_hetero = X + rng.randn(n_samples, n_features) * sigmas

###############################################################################
# Fit the models

n_components = np.arange(0, n_features, 5)  # options for n_components


def compute_scores(X):
    pca = PCA()
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X))


for X, title in [(X_homo, 'Homoscedastic Noise'),
                 (X_hetero, 'Heteroscedastic Noise')]:
    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    pca = PCA(n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
    plt.axvline(n_components_pca, color='b',
                label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r',
                label='FactorAnalysis CV: %d' % n_components_fa, linestyle='--')
    plt.axvline(n_components_pca_mle, color='k',
                label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

    # compare with other covariance estimators
    plt.axhline(shrunk_cov_score(X), color='violet',
                label='Shrunk Covariance MLE', linestyle='-.')
    plt.axhline(lw_score(X), color='orange',
                label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    plt.title(title)

plt.show()






# Dividend retrieval
ticker = 'GE'
start_date = dt.datetime(1990, 01, 01)
end_date = dt.datetime.today()
dividends = db.YahooApi.retrieve_dividends(equity_ticker=ticker, start_date=start_date, end_date=end_date)

# Options retrieval
above_below = 20
test_data = pdata.YahooOptions(ticker).get_all_data()

expiry_dates, links = pdata.YahooOptions(ticker)._get_expiry_dates_and_links()
expiry_dates = [date for date in expiry_dates if date >= dt.datetime.today().date()]


options_data = pdata.YahooOptions(ticker).get_near_stock_price(above_below=above_below, expiry=expiry_dates[1])
print('done')

options_data = db.YahooApi.retrieve_options_data('GOOGL')
print('done')


data = pdata.get_data_yahoo(symbols='^RUT',
                            start="01/01/2010",
                            end="05/01/2015")


returns = data['Adj Close'] / data['Adj Close'].shift(1) - 1

returns.index = returns.index.normalize()
if returns.index.tzinfo is None:
    returns.index = returns.index.tz_localize('UTC')

pyfolio.create_returns_tear_sheet(returns=returns, return_fig=True)





test_options = pdata.YahooOptions('AAPL').get_all_data()

plt.figure()
data['Adj Close'].plot()
plt.show()

opener = urllib.URLopener()
target = 'https://s3.amazonaws.com/static.quandl.com/tickers/SP500.csv'
opener.retrieve(target, 'SP500.csv')
tickers_file = pd.read_csv('SP500.csv')
tickers = tickers_file['ticker'].values

plt.figure()
plt.hist(returns[np.isfinite(returns)])