import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pandas_datareader.data as pdata
import pyfolio
import pymc
from matplotlib import cm

import qfl.core.calcs as calcs
import qfl.core.calcs as lib
import qfl.core.database_interface as qfl_data
import qfl.core.market_data as md
import qfl.core.constants as constants
import qfl.etl.data_ingest as etl
import qfl.etl.data_interfaces as data_int
import qfl.macro.macro_models as macro
import qfl.utilities.basic_utilities as utils
import qfl.core.portfolio_utils as putils

reload(calcs)
reload(utils)
reload(qfl_data)
reload(data_int)
reload(md)
reload(etl)

from qfl.etl.data_interfaces import QuandlApi, IBApi, DataScraper, NutmegAdapter, YahooApi
from qfl.core.database_interface import DatabaseInterface as db
from qfl.utilities.chart_utilities import format_plot
from qfl.utilities.nlp import DocumentsAnalyzer as da

from gensim import corpora, models, similarities


db.initialize()

reload(md)
from qfl.core.market_data import VolatilitySurfaceManager

vsm = VolatilitySurfaceManager()
vsm.load_data(tickers=['EWZ'])
tmp = vsm.get_surface_point(tickers=['EWZ'], call_delta=0.75, tenor_in_days=30)

# What if I need


'''
--------------------------------------------------------------------------------
Volatility relative value
--------------------------------------------------------------------------------
'''

# OK. Pure version: static strategy  would have PNL equal to
# gamma PNL plus term structure rolldown, e.g. it's as-if you owned
# an option that day

# Key thing here is that we have a PNL stream for every underlying
# So there's no "static strategy"
# I think I need 'portfolio RV strategy' or something

import qfl.strategies.volswap_rv as volswap_rv
reload(volswap_rv)
from qfl.strategies.volswap_rv import VolswapRvStrategy as vrv

vrv.initialize_data()
vrv.process_data()

rv_iv_signals = vrv.initialize_rv_iv_signals()
iv_signals = vrv.initialize_iv_signals()
ts_signals = vrv.initialize_ts_signals()

rv_iv_pnl, rv_iv_pos, rv_iv_pctile \
    = vrv.compute_signal_quantile_performance(signal_data=rv_iv_signals)

iv_pnl, iv_pos, iv_pctile \
    = vrv.compute_signal_quantile_performance(signal_data=iv_signals)

ts_pnl, ts_pos, ts_pctile \
    = vrv.compute_signal_quantile_performance(signal_data=ts_signals)

# signal_pnl, signal_pos, signal_pctile = vrv.compute_master_backtest()
print('done!')

# Save
vrv_data = vrv.data
vrv_settings = vrv.settings
vrv_calc = vrv.calc

# Restore
vrv.data = vrv_data
vrv.settings = vrv_settings
vrv.calc = vrv_calc

signal_pnl = rv_iv_pnl
signal_positions = rv_iv_pos
quantiles = signal_pnl.keys()
signal_com = np.unique(signal_pnl[1.0].columns.get_level_values(None))

buy_q = 0.80
sell_q = 0.20
tc_vega = 0.1

# Comparing COM for a long/short pair of quantiles
plt.figure()
for com in signal_com:
    l = signal_pnl[buy_q][com].sum(axis=1).cumsum()
    s = signal_pnl[sell_q][com].sum(axis=1).cumsum()
    tc = (signal_positions[buy_q][com].abs().sum(axis=1).cumsum()
       + signal_positions[sell_q][com].abs().sum(axis=1).cumsum()) * tc_vega
    plt.plot(l-s-tc)
plt.legend(signal_com, loc=2)

# Comparing quantiles for each COM
for com in signal_com:
    plt.figure()
    for i in range(1, len(quantiles)):
        plt.plot(signal_pnl[quantiles[i]][com].sum(axis=1).cumsum())
    plt.legend(quantiles[1:], loc=2)
    plt.title("com = " + str(com))

'''
--------------------------------------------------------------------------------
VIX curve
--------------------------------------------------------------------------------
'''

import qfl.strategies.strategies as strat
import qfl.strategies.vol_fut_curve as vft
reload(vft)
reload(strat)
from qfl.strategies.vol_fut_curve import VixCurveStrategy as vc

# Beginning of semi-clean VIX futures data
start_date = dt.datetime(2007, 3, 26)
holding_period_days = 1
signals_z_cap = 1.0
vol_target_com = 63
rolling_beta_com = 63

# Prep
vc.initialize_data(vol_futures_series='VX',
                   short_month=1,
                   long_month=5)

vc.compute_hedge_ratios(rolling_beta_com=rolling_beta_com)

# Main analysis
vc_output = vc.compute_master_backtest(
    holding_period_days=holding_period_days,
    signals_z_cap=signals_z_cap,
    vol_target_com=vol_target_com)

# Risk and return
vc_output.combined_pnl.std()
vc_output.combined_pnl['optim_weight'].rolling(21).sum().quantile(0.005)
vc_output.combined_pnl_net.rolling(21).sum().mean()

# Sensitivity analysis to weights
num_sims = 1000
sigma = 2.0
sens_percentile = 0.01
sim_perf_percentiles, sim_perf = \
    strat.PortfolioOptimizer.compute_signal_portfolio_sensitivity(
        strategy=vc,
        signals_data=vc_output.signal_output.signals_data,
        weights=vc_output.weights,
        num_sims=num_sims,
        sigma=sigma,
        signals_z_cap=signals_z_cap,
        holding_period_days=holding_period_days,
        vol_target_com=vol_target_com
    )

# Basic plot
plt.figure()
plt.plot(vc_output.combined_pnl_net['optim_weight'].cumsum())
plt.ylabel('cumulative strategy PNL, in vegas of max position size')
# plt.legend(['smart weighting', 'all signals equal weight'], loc=2)

# Momentum plot
plt.figure()
mom_signals = ['mom_5', 'mom_10', 'mom_21', 'mom_63']
plt.plot(vc_output.signal_output.signal_pnl[mom_signals].cumsum())
colormap = plt.get_cmap('coolwarm')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(mom_signals))])
plt.legend(mom_signals, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# Convexity plot
plt.figure()
cv_signals = ['cv_0', 'cv_1', 'cv_5', 'cv_10']
colormap = plt.get_cmap('spectral')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(cv_signals))])
plt.plot(vc_output.signal_output.signal_pnl[cv_signals].cumsum())
plt.legend(cv_signals, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# TS plot
plt.figure()
ts_signals = ['ts_0', 'ts_1', 'ts_5', 'ts_10']
colormap = plt.get_cmap('gist_heat')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(ts_signals))])
plt.plot(vc_output.signal_output.signal_pnl[ts_signals].cumsum())
plt.legend(ts_signals, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# Signal robustness plot
plt.figure()
plt.plot(vc_output.combined_pnl_net.cumsum())
plt.plot(sim_perf_percentiles.cumsum())
cols = ['optim_weight', 'equal_weight'] + [str(c) for c in sim_perf_percentiles.columns]
plt.legend(cols, loc=2)

'''
--------------------------------------------------------------------------------
Vega vs delta
--------------------------------------------------------------------------------
'''

import qfl.strategies.strategies as strat
import qfl.strategies.vega_vs_delta as vdm
reload(vdm)
from qfl.strategies.vega_vs_delta import VegaVsDeltaStrategy as vd
from qfl.core.database_interface import DatabaseUtilities as dbutils
from pandas.tseries.offsets import BDay

# Beginning of semi-clean VIX futures data
start_date = dt.datetime(2007, 3, 26)
change_window_days_short = 5
change_window_days_long = 252
holding_period_days = 1
signals_z_cap = 1.0
vol_target_com = 63
rolling_beta_com = 63
tc = 0.05

corr_r_shrinkage = 0.90
corr_er_shrinkage = 0.70
er_se_beta_to_er = 0.25
er_se_beta_to_vol = 0.75

vd.initialize_data()

vd_output = vd.compute_master_backtest(
    holding_period_days=holding_period_days,
    vol_target_com=vol_target_com,
    rolling_beta_com=rolling_beta_com,
    signals_z_cap=signals_z_cap,
    transaction_cost_per_unit=tc,
    corr_r_shrinkage=corr_r_shrinkage,
    corr_er_shrinkage=corr_er_shrinkage,
    signal_se_beta_to_er=er_se_beta_to_er,
    signal_se_beta_to_vol=er_se_beta_to_vol)

# Sensitivity analysis to weights
num_sims = 1000
sigma = 2.0
sens_percentile = 0.01

sim_perf_percentiles, sim_perf = \
    strat.PortfolioOptimizer.compute_signal_portfolio_sensitivity(
        strategy=vd,
        signals_data=vd_output.signal_output.signals_data,
        weights=vd_output.weights,
        num_sims=num_sims,
        sigma=sigma,
        signals_z_cap=signals_z_cap,
        holding_period_days=holding_period_days,
        vol_target_com=vol_target_com
    )


# Risk and return
vd_output.combined_pnl.rolling(21).sum().std()
vd_output.combined_pnl['optim_weight'].rolling(21).sum().iloc[21:].quantile(0.005)
vd_output.combined_pnl_net.rolling(21).sum().mean()
vd_output.combined_pnl_net.rolling(21).sum().mean() / vd_output.combined_pnl.rolling(21).sum().std() * np.sqrt(12)

plt.figure()
plt.plot(vd_output.combined_pnl_net['optim_weight'].cumsum(), color='k')
plt.plot(sim_perf_percentiles.cumsum())
plt.legend(['strategy', '1%', '10%', '25%', '50%', '75%', '90%', '99%'], loc=2)


# Basic plot
plt.figure()
plt.plot(vd_output.combined_pnl_net['optim_weight'].cumsum())
plt.ylabel('cumulative strategy PNL, in vegas of max position size')
# plt.legend(['smart weighting', 'all signals equal weight'], loc=2)

# Positioning plot
plt.figure()
pos_signals = ['vol_spec_pos', 'index_spec_pos',
               'vol_spec_pos_chg_s', 'index_spec_pos_chg_s',
               'vol_spec_pos_chg_l', 'index_spec_pos_chg_l']
plt.plot(vd_output.signal_output.signal_pnl[pos_signals].cumsum())
colormap = plt.get_cmap('coolwarm')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(pos_signals))])
plt.legend(pos_signals, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# RV  plot
plt.figure()
rv_signals = ['rv_10', 'rv_21', 'rv_42', 'rv_63']
colormap = plt.get_cmap('spectral')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(rv_signals))])
plt.plot(vd_output.signal_output.signal_pnl[rv_signals].cumsum())
plt.legend(rv_signals, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# TS plot
plt.figure()
ts_signals = ['ts_0', 'ts_1', 'ts_5', 'ts_10']
colormap = plt.get_cmap('gist_heat')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(ts_signals))])
plt.plot(vd_output.signal_output.signal_pnl[ts_signals].cumsum())
plt.legend(ts_signals, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# Weights comparison plot
plt.figure()
plt.plot(vd_output.combined_pnl_net.cumsum())
plt.ylabel('cumulative strategy PNL, in vegas of max position size')
plt.legend(['smart weighting', 'all signals equal weight'], loc=2)

# Signal robustness plot
plt.figure()
plt.plot(vd_output.combined_pnl_net.cumsum())
plt.plot(sim_perf_percentiles.cumsum())
cols = ['optim_weight', 'equal_weight'] + [str(c) for c in sim_perf_percentiles.columns]
plt.legend(cols, loc=2)
'''
--------------------------------------------------------------------------------
Evaluate predictiveness of past PNL for future PNL
--------------------------------------------------------------------------------
'''

# TODO: we should be weighting these inversely by correlation to each other
# Errors are no doubt correlated
# GLS?

# Here we're evaluating

com = 252
cap_z = 2.0

pnl_horizon = 63
com_range = [63, 126, 252, 512, 756, 1024]
cap_z_range = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

pnl_pred_result = pd.DataFrame(index=com_range, columns=cap_z_range)

for com in com_range:
    for cap_z in cap_z_range:

        pnl_pred_dataset = pd.DataFrame(calcs.windsorize(signal_pnl, cap_z)
                                        .ewm(com=com)
                                        .mean()
                                        .shift(1)
                                        .stack())

        pnl_pred_dataset = pnl_pred_dataset.rename(columns={0: 'pnl_ewm_lag'})
        pnl_pred_dataset['pnl'] = signal_pnl\
                                    .rolling(window=pnl_horizon)\
                                    .mean()\
                                    .shift(pnl_horizon)\
                                    .stack()

        # Only finite obs
        pnl_pred_dataset = pnl_pred_dataset[np.isfinite(pnl_pred_dataset)
            .all(axis=1)]

        # Filter out some noisy starting observations
        pnl_pred_dataset = pnl_pred_dataset[
            pnl_pred_dataset.index.get_level_values('date')
            > start_date + BDay(21)]

        pnl_pred_result.loc[com, cap_z] = \
            pnl_pred_dataset['pnl'].corr(pnl_pred_dataset['pnl_ewm_lag'])


'''
--------------------------------------------------------------------------------
Sensitivity analysis based on weights
--------------------------------------------------------------------------------
'''



'''
--------------------------------------------------------------------------------
Can we fit a reasonable model to vol and stocks and use that to bootstrap
--------------------------------------------------------------------------------
'''



'''
--------------------------------------------------------------------------------
ETF vol screen?
--------------------------------------------------------------------------------
'''

tickers, exchange_codes = md.get_etf_universe()
tickers = list(set(tickers) - set(['SHV', 'TIP', 'CIU', 'BKLN']))

iv = md.get_equity_implied_volatility(tickers=tickers,
                                      start_date=dt.datetime(2016, 9, 1))

screen = iv['iv_3mc'] / (iv['tick_rv_20d'] * 0.5 + iv['tick_rv_60d'] * 0.5)
screen = screen.sort_values()

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

plot_start_date = dt.datetime(2007, 1, 1)
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
req_url = "http://marketdata.websol.barchart.com/getQuote.json?key="
req_url += barchart.API_KEY
req_url += "&symbols=VIQ16"

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