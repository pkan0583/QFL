# execfile("startup.py")

import datetime as dt
import numpy as np
import pandas as pd
import struct
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sn

import qfl.core.calcs as calcs
import qfl.core.market_data as md
import qfl.core.constants as constants
import qfl.macro.macro_models as macro
import qfl.utilities.basic_utilities as utils

from qfl.core.database_interface import DatabaseInterface as db
from qfl.core.market_data import VolatilitySurfaceManager
from qfl.macro.macro_models import FundamentalMacroModel
import qfl.etl.data_ingest as etl

import qfl.models.volatility_factor_model as vfm
from qfl.models.volatility_factor_model import VolatilityFactorModel

import qfl.strategies.strategies as strat
from qfl.strategies.vix_curve import VixCurveStrategy
from qfl.strategies.equity_vs_vol import VegaVsDeltaStrategy
from qfl.strategies.vol_rv import VolswapRvStrategy
from qfl.strategies.strategies import PortfolioOptimizer
from qfl.strategies.strategy_master import StrategyMaster
import qfl.models.model_data_manager as mdm

from qfl.core.marking import SecurityMaster, MarkingEngine

import logging

db.initialize()
logger = logging.getLogger()


'''
--------------------------------------------------------------------------------
New centralized approach
--------------------------------------------------------------------------------
'''

start_date = dt.datetime(2010, 1, 1)
vsm = VolatilitySurfaceManager()
tickers = md.get_etf_vol_universe()
vsm.load_data(tickers=tickers, start_date=start_date)
# vsm.clean_data()
# clean_data = vsm.data

# Restore
clean_data = pd.read_excel('data/clean_ivol_data.xlsx')
clean_data['date'] = clean_data['date'].ffill()
clean_data = clean_data.set_index(['date', 'ticker'], drop=True)
vsm.data = clean_data
clean_data.to_excel('data/clean_ivol_data.xlsx')

# Strategy master
sm = StrategyMaster()

# VIX strategies
# sm.initialize_vix_curve(price_field='settle_price')
# sm.initialize_equity_vs_vol()

# Volatility RV
# sm.initialize_macro_model()
# sm.initialize_volatility_factor_model()
sm.initialize_volatility_rv(vsm=vsm)
model = sm.strategies['vol_rv']
signal_data = sm.outputs['vol_rv']['signal_data']
signal_name = 'rv_iv_10'

# Initialize
model.settings['start_date'] = dt.datetime(2010, 1, 5)
dates, trade_dates, maturity_dates, stock_prices, buys, sells, sizes \
    = model.initialize_portfolio_backtest(signal_data=signal_data,
                                          signal_name=signal_name)

# Create the security master
security_master, transactions, inputs, outputs \
    = model._create_sec_master_options(
        trade_dates=trade_dates,
        maturity_dates=maturity_dates,
        stock_prices=stock_prices,
        buys=buys,
        sells=sells,
        sizes=sizes,
        signal_data=signal_data
    )

# Create the positions
positions, portfolio_summary = \
    model._create_backtest_positions_options(security_master=security_master,
                                             transactions=transactions,
                                             inputs=inputs,
                                             outputs=outputs)
# Delta hedge
df = \
    (-positions['delta_shares'].unstack('instrument').shift(1)
    * positions['spot'].unstack('instrument').diff(1)
    * positions['quantity'].unstack('instrument')
     ).stack('instrument')
positions['delta_hedge_pnl'] = df.reset_index()\
    .set_index(['instrument', 'date'])[0]

portfolio_summary['delta_hedge_pnl'] = positions['delta_hedge_pnl']\
    .groupby(level='date').sum()
portfolio_summary['hedged_pnl'] = portfolio_summary['pnl'] \
                                + portfolio_summary['delta_hedge_pnl']
portfolio_summary['cum_pnl'] = portfolio_summary['hedged_pnl'].cumsum()


positions.to_excel('data/backtest_positions.xlsx')


# DEBUGGING

df = security_master.expand_attributes()
df_und = df[df['underlying'] == 'JNK']
mats = np.unique(pd.to_datetime(
    df_und['maturity_date'].values.tolist()).tolist())
strikes = np.unique(df_und['strike'].values.tolist())
tmp = vsm.get_fixed_maturity_date_vol_by_strike(
    ticker='JNK',
    strikes=strikes,
    maturity_dates=mats,
    start_date=start_date).stack(level=['strike', 'maturity_date'])

strikes = np.unique(strikes)
maturity_dates = np.unique(maturity_dates)

delta_grid = np.append([0.001, 0.01],
                       np.append(np.arange(0.05, 1.0, 0.05),
                                 [0.99, 0.999]))
ticker = 'JNK'
fm_data = vsm.get_data(tickers=[ticker],
                        start_date=start_date)

tenor_string = 'days_to_maturity_' + str(
    3) + 'mc'
tenor_in_days = fm_data[tenor_string] \
    .reset_index(level='ticker', drop=True)
fm_vols = vsm.get_surface_point_by_delta(
    tickers=[ticker],
    call_delta=delta_grid.tolist(),
    tenor_in_days=tenor_in_days,
    start_date=start_date).reset_index(level='ticker', drop=True)








strategy_versions = sm.get_strategy_versions()
# backtest_update_start_date = dt.datetime(2016, 10, 1)
# etl.ingest_strategy_backtest_data(load_clean_data=True)

strategy_name = 'vol_rv'
model = sm.strategies[strategy_name]

# Long-only, short-only and long/short versions

model.settings = strategy_versions['portfolio']['vol_rv']['default']
d_n = mdm.get_model_outputs(model_name=model.name, settings=model.settings)
d_n = d_n[d_n['output_name'] == 'pnl_gross'].set_index('date')['value']

model.settings = strategy_versions['portfolio']['vol_rv']['short_only']
d_s = mdm.get_model_outputs(model_name=model.name, settings=model.settings)
d_s = d_s[d_s['output_name'] == 'pnl_gross'].set_index('date')['value']

model.settings = strategy_versions['portfolio']['vol_rv']['long_only']
d_l = mdm.get_model_outputs(model_name=model.name, settings=model.settings)
d_l = d_l[d_l['output_name'] == 'pnl_gross'].set_index('date')['value']

df = pd.DataFrame()
df['rv'] = d_n
df['short'] = d_s
df['long'] = d_l

signal_pnl = mdm.get_portfolio_strategy_signal_pnl(model=model)
signal_data = sm.outputs[strategy_name]['signal_data']

# Run the optimization and the final backtest
positions, portfolio_summary, sec_master, optim_output = \
    model.compute_master_backtest(
        signal_pnl=signal_pnl,
        signal_data=signal_data,
        backtest_update_start_date=start_date
    )

mdm.archive_portfolio_strategy_positions(model=model, positions=positions)



call_deltas_initial = [0.40, 0.60]
model.compute_portfolio_backtest_options(
    signal_data=signal_data,
    signal_name=signal,
    call_deltas_initial_position=call_deltas_initial)












'''
--------------------------------------------------------------------------------
strategy pnl analysis
--------------------------------------------------------------------------------
'''

vrv_outputs = mm.get_model_outputs(model_name='vol_rv',
                                   settings=sm.strategies['vol_rv'].settings)
# annoying: i don't have this archived yet
s = "select date, pnl, signal_name from model_signal_data_view " \
    "where model_name = 'vol_rv' " \
    "and ref_entity_id = 'strategy' and signal_name in ('ts_1', 'rv_iv_21')"
vrv_pnl = db.read_sql(s)
vrv_pnl = vrv_pnl.set_index(['date', 'signal_name'])['pnl']\
    .unstack('signal_name')
vrv_pnl['weighted'] = -vrv_pnl['ts_1'] / 2 + vrv_pnl['rv_iv_21'] / 2



# Returns versus risk factors
strategy_name = 'equity_vs_vol'
df = pd.DataFrame()
return_windows = [1, 5, 10, 21, 63]

df['returns'] = underlying_price / underlying_price.shift(t1) - 1
df['pnl'] = pd.to_numeric(sm.outputs[strategy_name]['combined_pnl_net']
                          ['optim_weight']).rolling(t1).sum()
df = df[np.isfinite(df).all(axis=1)]
plt.scatter(y=df['pnl'], x=df['returns'])


etl.add_equities_from_list(tickers=['IEI'], exchange_codes=['US'])
ids = db.get_equity_ids(equity_tickers=['IEI'])
etl.ingest_historical_equity_prices(ids=ids)

'''
--------------------------------------------------------------------------------
Bayesian time-varying expected returns
--------------------------------------------------------------------------------
'''

r = sm.outputs['vix_curve']['combined_pnl_net']['optim_weight']
r = r[np.isfinite(pd.to_numeric(r))]

psi_bar = 0.0
rho = 0.998
sig2_eps = r.var()
sig2_eta = sig2_eps / 252.0 / 2.0
sig2_psi0 = sig2_eta

lmda = np.ones(len(r))

num_sims = 100

psi_shocks = np.random.randn(len(r), num_sims)
psi_shocks[:, num_sims/2:] = -psi_shocks[:, 0:num_sims/2]

psi_sim = np.zeros([len(r), num_sims])
psi_sim[0, :] = psi_bar + sig2_psi0 ** 0.5 * psi_shocks[0, :]
psi_mean = np.zeros([len(r), num_sims])
psi_var = np.zeros(len(r))
for t in range(1, len(r)):
    psi_var[t] = 1.0 / (lmda[t] / sig2_eps + 1.0 / sig2_eta)
    psi_mean[t, :] = psi_var[t] * (r.iloc[t] / (sig2_eps / lmda[t])
        + (rho * psi_sim[t-1, :] + psi_bar * (1 - rho)) / sig2_eta)
    psi_sim[t, :] = (psi_mean[t, :] + psi_shocks[t, :] * sig2_eta ** 0.5)

psi_df = pd.DataFrame(index=r.index, data=psi_sim)

tmp = pd.DataFrame(index=r.index, data=psi_mean.mean(axis=1))
plt.plot(tmp)



plt.plot(r.cumsum())

# Objective is to get iterative method for density of psi(t)
# This is the density of psi(t) at psi(t-1)=x
def psi_t_dens(x, params):
    a = (1.0 / np.sqrt(2 * np.pi * params['sig2_eta']))
    b = np.exp(-(x - params['rho'] * params['psi_t-1'] - params['psi_bar']) ** 2.0
               / (2.0 * params['sig2_eta']))
    c = (1.0 / np.sqrt(2 * np.pi * params['sig2_psi0']))
    d = np.exp(-(params['psi_t-1'] - params['psi_bar']) ** 2.0 / (2.0 * params['sig2_psi']))
    return a * b * c * d


# This is the density of psi(t-1) at psi(t) given
def psi_dens(x, params):
    a = (1.0 / np.sqrt(2 * np.pi * params['sig2_eta']))
    b = np.exp(-(params['psi_t'] - params['rho'] * x - params['psi_bar']) ** 2.0
               / (2.0 * params['sig2_eta']))
    c = (1.0 / np.sqrt(2 * np.pi * params['sig2_psi0']))
    d = np.exp(-(x - params['psi_bar']) ** 2.0 / (2.0 * params['sig2_psi0']))
    return a * b * c * d

from scipy.integrate import quad
params = dict()
params['sig2_eta'] = sig2_eta
params['rho'] = rho
params['sig2_psi'] = sig2_psi0
params['psi_bar'] = psi_bar

# example of that density at a point (for psi_t-1)
params['psi_t'] = 0.01
psi_dens(params['psi_t'], params)

# drawing the density over a grid
grid = np.arange(-0.1, 0.1, 0.001)
tmp = pd.DataFrame(index=grid, data=psi_dens(x=grid, params=params))
plt.plot(tmp)

# example of that density at a point, integrating out




###################################
# Aside: what about these quantiles
###################################

# Remember that the idea here is every day getting the PNL of one day of a
# constant-maturity 3-month volswap
# So the fair transactino cost is 1/63 of one

buy_q = 0.80
sell_q = 0.40
tc_vega = 0.5 / 63.0 * 2.0

signal_pnl, signal_positions, signal_pctile = \
    sm.strategies['vol_rv'].compute_signal_quantile_performance(
        signal_data=signal_data)
total_signal_pnl = pd.DataFrame(columns=signal_data.columns)
total_signal_pnl_net = pd.DataFrame(columns=signal_data.columns)
for signal in signal_data.columns:
    direction = 1.0
    if "iv" in signal or "ts" in signal and "iv_rv" not in signal:
        direction = -1.0
    total_signal_pnl[signal] = (signal_pnl[buy_q][signal].sum(axis=1)
                             - signal_pnl[sell_q][signal].sum(axis=1)) \
                             / signal_positions[buy_q][signal].sum(axis=1)
    total_signal_pnl[signal] *= direction
    total_signal_pnl_net[signal] = total_signal_pnl[signal] \
                                   - tc_vega * np.ones(len(total_signal_pnl))
print('done!')

# Question: RV by itself seems negative here?
signal_group = 'ts'
cols = [s for s in signal_data.columns if signal_group in s and 'rv_iv' not in s]
plt.plot(total_signal_pnl_net[cols].cumsum())
plt.legend(cols, loc=2)





# Current VIX term structure
cols = ['days_to_maturity', 'settle_price', 'seasonality_adj_price']
vix_spot = md.get_generic_index_prices(tickers=['VIX'])
vix_ts = md.get_futures_prices_by_series(futures_series='VX')[cols]
vix_ts['seasonality_adj_price'] *= np.sqrt(20.0/21.0)
vix_ts.loc[(md.market_data_date, 'VIX'), cols] = [0, 13.29, 13.29]
vix_ts = vix_ts.sort_values('days_to_maturity')
plt.plot(vix_ts.set_index(['days_to_maturity'], drop=True))


# Latest signals
sm.outputs['vix_curve']['signal_data'].loc[md.market_data_date]
sm.outputs['vix_curve']['signal_data_z'].loc[md.market_data_date]
sm.outputs['vix_curve']['optim_output']['weights']
sm.outputs['vix_curve']['positions']

# Latest signals
sm.outputs['equity_vs_vol']['signal_data'].loc[md.market_data_date]
sm.outputs['equity_vs_vol']['signal_data_z'].loc[md.market_data_date]
sm.outputs['equity_vs_vol']['optim_output']['weights']
sm.outputs['equity_vs_vol']['positions']




'''
--------------------------------------------------------------------------------
Volatility relative value
--------------------------------------------------------------------------------
'''

###############################################################################
# Inputs (macro model)
###############################################################################
#
# Rolling window for factor model
minimum_obs = 21
window_length_days = 512
update_interval_days = 21
iv_com = 63
n_components = 3
factor_model_start_date = dt.datetime(2010, 1, 1)

# Getting some data
tickers = md.get_etf_vol_universe()
#
vsm = VolatilitySurfaceManager()
vsm.load_data(tickers=tickers, start_date=factor_model_start_date)
VolatilityFactorModel.initialize_data(vsm=vsm,
                                      tickers=tickers,
                                      iv_com=iv_com)

factor_weights_composite, factor_data_composite, factor_data_oos = \
    VolatilityFactorModel.run(minimum_obs=minimum_obs,
                              window_length_days=window_length_days,
                              update_interval_days=update_interval_days,
                              n_components=n_components)

factor_weights_insample, factor_data_insample = \
    VolatilityFactorModel.run_insample(n_components=n_components)

# One-liner for fundamental model
macro_output = FundamentalMacroModel.run()

# Keep in mind that these can't really be observed at time (t)
# More like 1-2 months after
macro_factors = macro_output.pca_factors
macro_factors_ma = macro_output.pca_factors_ma


###############################################################################
# Outputs (VRV)
###############################################################################

reload(md)
from qfl.core.market_data import VolatilitySurfaceManager
vsm = VolatilitySurfaceManager()
vsm.load_data(tickers, start_date=factor_model_start_date)


import qfl.strategies.vol_rv as vrv
reload(vrv)
vrv = VolswapRvStrategy()
vrv.initialize_data(volatility_surface_manager=vsm)
vrv.process_data()

# Save
vrv_data = vrv.strat_data
vrv_settings = vrv.settings
vrv_calc = vrv.calc

# Restore
vrv.strat_data = vrv_data
vrv.settings = vrv_settings
vrv.calc = vrv_calc
vrv.set_universe()

rv_iv_signals = vrv.initialize_rv_iv_signals()
iv_signals = vrv.initialize_iv_signals()
ts_signals = vrv.initialize_ts_signals()
rv_signals = vrv.initialize_rv_signals()
tick_rv_signals = vrv.initialize_tick_rv_signals()

macro_signals = vrv.initialize_macro_signals(
    factor_weights=factor_weights_insample,
    macro_factors=macro_factors_ma)

for signal in macro_signals:
    macro_signals[signal] = macro_signals[signal]\
                            .unstack('ticker')\
                            .fillna(method='ffill')\
                            .stack('ticker')

signal_data = pd.concat([rv_iv_signals,
                         iv_signals,
                         ts_signals,
                         rv_signals,
                         tick_rv_signals], axis=1)

# Actually compute that stuff (very slow)
# portfolio_summary = dict()
# positions = dict()
#
# for signal in signal_data.columns:
#
#     sd = pd.DataFrame(signal_data[signal])
#
#     positions, portfolio_summary[signal] = \
#         vrv.compute_realistic_portfolio_backtest(signal_data=signal_data)
#
# portfolio_summary_df = pd.concat(portfolio_summary)
#


###############################################################################
# Load the result and analyze
###############################################################################

# Raw PNL data
portfolio_summary_df = pd.read_excel('volatility_rv.xlsx').reset_index()
portfolio_summary_df['index'] = portfolio_summary_df['index'] \
                                .fillna(method='ffill')
portfolio_summary_df = portfolio_summary_df.set_index(['index', 'date'])

# Unstacking
daily_pnl = portfolio_summary_df['total_pnl'].unstack('index')
pnl = portfolio_summary_df['cum_pnl'].unstack('index')
gross_vega_long = portfolio_summary_df['gross_vega'].unstack('index') / 2.0

# Adjust for sign
for signal in ts_signals.columns + iv_signals.columns:
    daily_pnl[signal] *= -1.0
    pnl[signal] *= -1.0
more_signals = ['vf_0-mf_0']
for signal in more_signals:
    daily_pnl[signal] *= -1.0
    pnl[signal] *= -1.0

macro_signal_map = {'vf_0-mf_0': 'growth --> high vol beta vs low vol beta',
                    'vf_0-mf_1': 'inflation --> high vol beta vs low vol beta',
                    'vf_1-mf_0': 'growth --> old economy vs new economy',
                    'vf_1-mf_1': 'inflation --> old economy vs new economy',
                    'vf_2-mf_0': 'growth --> ??',
                    'vf_2-mf_1': 'inflation --> vs yield'}

# Macro growth
plt.figure()
plt.plot(pnl[['vf_0-mf_0', 'vf_1-mf_0', 'vf_2-mf_0']] / gross_vega_long['vf_0-mf_0'].mean())
plt.legend(['F1 (vol beta)', 'F2 (old vs new economy)', 'F3 (yield)'], loc=2)
plt.ylabel('performance, multiple of long vega')
#
# # Macro inflation
plt.figure()
plt.plot(pnl[['vf_0-mf_1', 'vf_1-mf_1', 'vf_2-mf_1']] / gross_vega_long['vf_0-mf_1'].mean())
plt.legend(['F1 (vol beta)', 'F2 (old vs new economy)', 'F3 (yield)'], loc=2)
plt.ylabel('performance, multiple of long vega')

# Realized/implied
plt.figure()
signal_names = ['model1', 'model2', 'model3', 'model4', 'model5']
for signal in rv_iv_signals.columns:
    if signal != 'rv_iv_252':
        plt.plot(pnl[signal] / gross_vega_long[signal].mean())
plt.legend(signal_names, loc=2)
plt.ylabel(('cumulative PNL, multiple of average long vega'))

# Implied
plt.figure()
signal_names = ['shortest', 'short', 'mid', 'long', 'longest']
for signal in iv_signals.columns:
    plt.plot(pnl[signal] / gross_vega_long[signal].mean())
plt.legend(iv_signals.columns, loc=2)
plt.ylabel(('cumulative PNL, multiple of average long vega'))

# Slope
plt.figure()
signal_names = ['method1', 'method2', 'method3', 'method4', 'method5']
for signal in ts_signals.columns:
    plt.plot(pnl[signal] / gross_vega_long[signal].mean())
plt.legend(signal_names, loc=2)
plt.ylabel(('cumulative PNL, multiple of average long vega'))

# RV
plt.figure()
signal_names = ['model1', 'model2', 'model3', 'model4', 'model5']
for signal in rv_signals.columns:
    if signal != 'rv_10':
        plt.plot(pnl[signal] / gross_vega_long[signal].mean())
plt.legend(signal_names, loc=2)
plt.ylabel(('cumulative PNL, multiple of average long vega'))

# TickRV
plt.figure()
signal_names = ['model1', 'model2', 'model3', 'model4', 'model5']
for signal in tick_rv_signals.columns:
    plt.plot(pnl[signal] / gross_vega_long[signal].mean())
plt.legend(signal_names, loc=2)
plt.ylabel(('cumulative PNL, multiple of average long vega'))


optim_weight_pnl_net = optim_weight_pnl_gross['composite'] - tc_ts['cum_tc']
optim_weight_pnl_net = optim_weight_pnl_net.fillna(method='ffill')
plt.plot(equal_weight_pnl_net / gross_vega_long.mean().mean())
plt.plot(optim_weight_pnl_net / gross_vega_long.mean().mean(), color='g')
plt.ylabel(('cumulative PNL, multiple of average long vega'))
plt.legend(['naive equal signal weight', 'robust Bayesian weight'], loc=2)

# Return profile versus S&P
spx = md.get_equity_prices(tickers=['SPY'],
                           start_date=vrv.settings.start_date)\
                            .unstack('ticker')['adj_close']
plot_data = 100.0 * (spx / spx.shift(1) - 1)

plot_data['strategy_pnl'] = equal_weight_pnl_gross.diff(1) / gross_vega_long.mean().mean()
plot_data = plot_data[np.isfinite(plot_data).all(axis=1)]

# Scatterplot of returns profile versus S&P monthly returns
fig, axarr = plt.subplots(1, 2)
fmt = '%.0f%%'
xticks = mtick.FormatStrFormatter(fmt)

axarr[0].scatter(plot_data['SPY'], plot_data['strategy_pnl'],
                 c=plot_data.index, cmap=cm.coolwarm)
axarr[0].set_title('Strategy returns versus S&P, daily')
axarr[0].set_ylabel('PNL, multiple of long vega')
axarr[0].set_xlabel('S&P 500 return')
axarr[0].xaxis.set_major_formatter(xticks)

plot_data = plot_data.resample('M').sum()
axarr[1].scatter(plot_data['SPY'], plot_data['strategy_pnl'],
                 c=plot_data.index, cmap=cm.coolwarm)
axarr[1].set_title('Strategy returns versus S&P, monthly')
axarr[1].set_ylabel('PNL, multiple of long vega')
axarr[1].set_xlabel('S&P 500 return')
axarr[1].xaxis.set_major_formatter(xticks)

###############################################################################
# Sensitivity analysis to weights
###############################################################################

num_sims = 1000
sigma = 2.0
sens_percentile = 0.01
equal_weights = pd.DataFrame(index=included_signals,
                             data=np.ones(len(included_signals))
                                  / len(included_signals),
                             columns=['weight'])
sim_perf_percentiles, sim_perf = \
    strat.PortfolioOptimizer.compute_rv_signal_portfolio_sensitivity(
        signals_pnl=daily_pnl[included_signals],
        weights=optim_output.weights,
        num_sims=num_sims,
        sigma=sigma
    )

# Signal robustness plot: equal/optim weight
plt.figure()
plt.plot(optim_weight_pnl_gross / gross_vega_long.mean().mean(), color='g')
plt.plot(equal_weight_pnl_gross / gross_vega_long.mean().mean(), color='k')

# Signal robustness plot: percentiles
colormap = plt.get_cmap('coolwarm')
plt.gca().set_color_cycle([colormap(i)
    for i in np.linspace(0, 0.9, len(sim_perf_percentiles.columns))])
plt.plot(sim_perf_percentiles.cumsum())
cols = ['optim_weight'] + \
       [str(c) for c in sim_perf_percentiles.columns]
plt.legend(cols, loc=2)


###############################################################################
# playing with the idea of an option strategy
###############################################################################

# reload(md)
# from qfl.core.market_data import VolatilitySurfaceManager
# vsm = VolatilitySurfaceManager()
# vsm.load_data(tickers=md.get_etf_vol_universe(), start_date=dt.datetime(2010, 1, 1))

vrv = sm.strategies['vol_rv']

trade_frequency_days = 5
trade_tenor_month = 3
quantiles = [0.10, 0.30, 0.70, 0.90, 1.0]
buy_q = 0.90
sell_q = 0.30
hedge_ratio_vov_weight = 0.50

call_deltas_initial_position = [0.40, 0.60]
relative_sizes_initial_position = [1.0, 1.0]
vrv_data = vrv.strat_data['vrv_data']

signal_name = 'rv_iv_21'
signal_data = pd.DataFrame(sm.outputs['vol_rv']['signal_data'][signal_name])

sec_cols = ['instrument',
            'underlying',
            'trade_date',
            'maturity_date',
            'strike',
            'option_type',
            'quantity']

# Template data
start_date = vrv.settings['start_date']
sample_data = vsm.get_data(tickers=['SPY'],
                           start_date=start_date)
dates = sample_data.index.get_level_values('date').sort_values()

maturity_dates = vsm.get_roll_schedule(tickers=['SPY'],
                                       start_date=vrv.settings['start_date'])['SPY']

# Stock prices
stock_prices = md.get_equity_prices(tickers=tickers,
                                    price_field='adj_close',
                                    start_date=start_date)\
                                    ['adj_close']

# Trade dates
trade_ind = np.arange(0, len(dates), trade_frequency_days)
trade_dates = sample_data \
    .iloc[trade_ind] \
    .sort_index(level='date') \
    .index.get_level_values('date')

# Use the "ideal" positions logic
signal_ideal_pnl, signal_ideal_positions, signal_pctile \
    = vrv.compute_signal_quantile_performance(signal_data=signal_data,
                                              quantiles=quantiles)

# Identify buys and sells
buys = signal_ideal_positions[buy_q][signal_name].loc[trade_dates]
sells = signal_ideal_positions[sell_q][signal_name].loc[trade_dates]

# Size the trades
avg_vol_of_vol = vrv_data['vol_of_iv_3m'].mean().mean()
sizes = ((1 - hedge_ratio_vov_weight) + hedge_ratio_vov_weight
         * avg_vol_of_vol / vrv_data['vol_of_iv_3m']) \
    .unstack('ticker')

# Scalar
c = constants.trading_days_per_month * (trade_tenor_month - 1)
sec_dict_t = dict()
t = 0

# Fill forward the VSM data
vsm.data = vsm.data.unstack('ticker').fillna(method='ffill', limit=5).stack('ticker')
stock_prices = stock_prices.unstack('ticker').fillna(method='ffill', limit=5).stack('ticker')

for trade_date in trade_dates:

    print(trade_date)

    # Identify the buys and sells
    trade_date_buys = (buys.loc[trade_date][
                           buys.loc[trade_date] > 0]).reset_index()
    trade_date_sells = (sells.loc[trade_date][
                            sells.loc[trade_date] > 0]).reset_index()

    if len(trade_date_buys) == 0:
        continue

    # Create the security master structure
    num_trades = len(trade_date_buys) + len(trade_date_sells)
    trade_ids = np.arange(0, num_trades)
    securities = pd.DataFrame(index=trade_ids, columns=sec_cols)
    securities['trade_date'] = trade_date

    # Underlying
    buy_ind = range(0, len(trade_date_buys))
    sell_ind = range(len(trade_date_buys), num_trades)
    securities.loc[buy_ind, 'underlying'] = trade_date_buys[
        'ticker'].values
    securities.loc[sell_ind, 'underlying'] = trade_date_sells[
        'ticker'].values
    underlyings = [str(s) for s in securities['underlying'].tolist()]

    # Relevant data
    td_data = vsm.get_data(tickers=securities['underlying'].tolist(),
                           start_date=trade_date,
                           end_date=trade_date) \
                           .reset_index(level='date', drop=True)

    sp_data = stock_prices[
        (stock_prices.index.get_level_values('date') == trade_date)
      & (stock_prices.index.get_level_values('ticker').isin(underlyings))
       ].reset_index(level='date', drop=True)

    # Traded sizes
    trade_date_sizes = sizes.loc[trade_date]
    securities.loc[buy_ind, 'quantity'] = trade_date_sizes[
        buys.loc[trade_date] > 0].values
    securities.loc[sell_ind, 'quantity'] = -trade_date_sizes[
        sells.loc[trade_date] > 0].values

    # Temporarily reset index to underlying in order to map to other data
    securities = securities.reset_index().set_index('underlying', drop=True)

    # Maturity dates
    maturity_date_fieldname = 'maturity_date_' + str(trade_tenor_month) + 'mc'
    securities['maturity_date'] = td_data[maturity_date_fieldname]
    tenor_in_days = utils.networkdays(start_date=securities['trade_date'],
                                      end_date=securities['maturity_date'])

    # Now loop over the delta range to get all the individual options
    sec_dict = dict()
    for call_delta in call_deltas_initial_position:

        vols = vsm.get_surface_point_by_delta(tickers=underlyings,
                                              call_delta=call_delta,
                                              tenor_in_days=tenor_in_days,
                                              start_date=trade_date,
                                              end_date=trade_date)\
                                     .reset_index(level='date', drop=True)

        sec_dict[call_delta] = securities.copy()
        sec_dict[call_delta]['ivol_0'] = vols

        # Get moneyness from delta

        # TODO: implement dividends and risk-free rates
        # TODO: better rounding
        sec_dict[call_delta]['strike'] = (calcs.black_scholes_moneyness_from_delta(
            call_delta=call_delta,
            tenor_in_days=tenor_in_days,
            ivol=vols / 100.0,
            risk_free=0,
            div_yield=0
        ) * sp_data).round(2)

    # Collapse back and overwrite
    securities = pd.concat(sec_dict).reset_index().rename(columns={'level_0':
                                                                   'delta_0'})

    # Name the options
    put_ind = securities.index[securities['delta_0'] > 0.50]
    call_ind = securities.index[securities['delta_0'] <= 0.50]
    securities.loc[put_ind, 'instrument'] = \
        securities['underlying'].map(str) + " " \
        + securities['maturity_date'].dt.date.map(str) + " P" \
        + securities['strike'].map(str)
    securities.loc[call_ind, 'instrument'] = \
        securities['underlying'].map(str) + " "  \
        + securities['maturity_date'].dt.date.map(str) + " C" \
        + securities['strike'].map(str)
    securities.loc[put_ind, 'option_type'] = 'p'
    securities.loc[call_ind, 'option_type'] = 'c'
    securities = securities.set_index('instrument', drop=True)

    t += 1
    sec_dict_t[trade_date] = securities

sec_df = pd.concat(sec_dict_t)

# OK. Intermediate data.
# What is the most efficient way to get implied vols for this portfolio?
# I've got a bunch of moneynesses for each underlier on each day

# I should really set these into an intermediate "option price database"
# and create their names. That's better

# Group by underlying
unique_und = np.unique(sec_df['underlying'])
call_dict = dict()
put_dict = dict()

# For the full period
for und in unique_und:

    print(und)

    sec = sec_df[sec_df['underlying'] == und]\
                .reset_index()\
                .set_index(['delta_0', 'trade_date'])\
                .unstack('delta_0')
    del sec['level_0']

    sp = pd.DataFrame(stock_prices.unstack('ticker')[und])

    all_strikes = np.unique(sec.stack('delta_0')['strike'])

    fm_data = vsm.get_data(tickers=['SPY'],
                           start_date=start_date)
    fm_data = fm_data[['iv_1mc', 'iv_2mc', 'iv_3m', 'iv_4mc',
                       'days_to_maturity_1mc', 'days_to_maturity_2mc',
                       'days_to_maturity_3mc', 'days_to_maturity_4mc',
                       'skew', 'curvature', 'skew_inf', 'curvature_inf']]
    tenor_string = 'days_to_maturity_' + str(
        trade_tenor_month) + 'mc'
    tenor_in_days = fm_data[tenor_string] \
        .reset_index(level='ticker', drop=True)

    vol_by_strike = vsm.get_fixed_maturity_vol_by_strike(
        ticker='SPY',
        strikes=all_strikes,
        contract_month_number=trade_tenor_month,
        start_date=start_date
        ) / 100.0

    dates = vol_by_strike.index

    tenor_matrix = np.tile(tenor_in_days.loc[dates], [len(all_strikes), 1]).transpose()
    spot_matrix = np.tile(sp.loc[dates], [1, len(all_strikes)])
    strike_matrix = np.tile(all_strikes.transpose(), [len(dates), 1])

    for strike in all_strikes:
        vol_by_strike[strike] = pd.to_numeric(vol_by_strike[strike])

    call_px = calcs.black_scholes_price(spot=spot_matrix,
                                        strike=strike_matrix,
                                        tenor_in_days=tenor_matrix,
                                        ivol=vol_by_strike,
                                        div_amt=0.0,
                                        risk_free=0.0,
                                        option_type='c')

    put_px = calcs.black_scholes_price(spot=spot_matrix,
                                       strike=strike_matrix,
                                       tenor_in_days=tenor_matrix,
                                       ivol=vol_by_strike,
                                       div_amt=0.0,
                                       risk_free=0.0,
                                       option_type='p')

    call_dict[und] = pd.DataFrame(index=dates,
                                  columns=all_strikes,
                                  data=call_px)\
                                  .stack()

    put_px = calcs.put_call_parity(input_price=call_px,
                                   option_type='c',
                                   strike=strike_matrix,
                                   spot=spot_matrix,
                                   tenor_in_days=tenor_matrix,
                                   div_amt=0.02,
                                   risk_free=0.0)
    put_dict[und] = pd.DataFrame(index=dates,
                                 columns=all_strikes,
                                 data=put_px)\
                                 .stack()

put_px_df = pd.concat(put_dict)
call_px_df = pd.concat(call_dict)

put_px_df.index.names = ['ticker', 'date', 'strike']
call_px_df.index.names = ['ticker', 'date', 'strike']

sec_dict_t = sec_dict_t.sort()


position_dates = stock_prices.unstack('ticker')\
                    .index.get_level_values(level='date')
all_options = np.unique(sec_df.index.get_level_values('instrument'))
positions = pd.DataFrame(index=position_dates, columns=all_options)

# Purchase prices
sec_df['price_0'] = np.nan
for trade_date in trade_dates:

    sec = sec_df[sec_df['trade_date'] == trade_date]

    # Bucket by maturity date
    mtds = np.unique(sec['maturity_date'])
    for mtd in mtds:

        # Note: it would be better to unify the option prices

        call_mtd = sec[(sec['maturity_date'] == mtd) & (sec['delta_0'] < 0.50)]\
            .reset_index().rename(columns={'level_0': 'date'})

        put_mtd = sec[(sec['maturity_date'] == mtd) & (sec['delta_0'] > 0.50)]\
            .reset_index().rename(columns={'level_0': 'date'})

        pp = put_px_df[put_px_df.index.get_level_values('date') == trade_date] \
            .reset_index().rename(columns={'ticker': 'underlying', 0: 'price'})
        put_mtd = pd.merge(left=put_mtd,
                           right=pp,
                           on=['date', 'underlying', 'strike'])
        put_mtd = put_mtd.set_index(['date', 'instrument'])
        sec_df.loc[put_mtd.index, 'price_0'] = put_mtd['price']

        cp = call_px_df[call_px_df.index.get_level_values('date') == trade_date] \
            .reset_index().rename(columns={'ticker': 'underlying', 0: 'price'})
        call_mtd = pd.merge(left=call_mtd,
                            right=pp,
                            on=['date', 'underlying', 'strike'])
        call_mtd = call_mtd.set_index(['date', 'instrument'])
        sec_df.loc[call_mtd.index, 'price_0'] = call_mtd['price']

# OK now create positions

port_cols = ['instrument',
             'date',
             'quantity',
             'tenor_in_days',
             'iv',
             'rv',
             'market_value',
             'daily_pnl']

t = 1

date = dates[t]

sec = securities

pos = pd.DataFrame(index=sec.index, columns=port_cols)
pos['date'] = date
for col in sec_cols:
    if col in sec.columns:
        pos[col] = sec[col]

pos['tenor_in_days'] = utils.networkdays(
    start_date=sec['trade_date'],
    end_date=sec['maturity_date'])

sp_data = stock_prices[
    (stock_prices.index.get_level_values('date') == date)
  & (stock_prices.index.get_level_values('ticker').isin(underlyings))
   ].reset_index(level='date', drop=True)
sp_data.index.names = ['underlying']

pos = pos.join(sp_data, on='underlying')\
    .rename(columns={'adj_close': 'spot_price'})

moneyness = pos['strike'] / pos['spot_price']

unique_und = np.unique(pos['underlying']).tolist()
for und in unique_und:

    vrv._vsm.get_data(tickers=[und], start_date=date, end_date=date)

'''
--------------------------------------------------------------------------------
VIX curve plots
--------------------------------------------------------------------------------
'''

vc_output = sm.outputs['vix_curve']
vc = sm.strategies['vix_curve']

# Basic plot
plt.figure()
plt.plot(vc_output['combined_pnl_net']['optim_weight'].cumsum())
plt.ylabel('cumulative strategy PNL, in vegas of max position size')
# plt.legend(['smart weighting', 'all signals equal weight'], loc=2)

# Momentum plot
plt.figure()
mom_signals = ['mom_5', 'mom_10', 'mom_21', 'mom_63']
plt.plot(vc_output['signal_pnl'][mom_signals].cumsum())
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
plt.plot(vc_output['signal_pnl'][cv_signals].cumsum())
plt.legend(cv_signals, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# TS plot
plt.figure()
ts_signals = ['ts_0', 'ts_1', 'ts_5', 'ts_10']
colormap = plt.get_cmap('gist_heat')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(ts_signals))])
plt.plot(vc_output['signal_pnl'][ts_signals].cumsum())
plt.legend(ts_signals, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# Impact of smart weighting
plt.figure()
plt.plot(vc_output['combined_pnl_net'].cumsum())
plt.ylabel('cumulative strategy PNL, in vegas of max position size')
plt.legend(['bayesian weighting', 'naive equal weight'], loc=2)

# Signal robustness plot
plt.figure()
plt.plot(vc_output['combined_pnl_net']['optim_weight'].cumsum(), color='k')
plt.plot(vc_output['combined_pnl_net']['equal_weight'].cumsum(), color='g')
colormap = plt.get_cmap('coolwarm')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9,
                                len(sim_perf_percentiles.columns))])
plt.plot(sim_perf_percentiles.cumsum())
plt.plot(sim_perf_percentiles.cumsum())
cols = ['optim_weight', 'equal_weight'] + [str(c)
                            for c in sim_perf_percentiles.columns]
plt.legend(cols, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# Scatterplot of returns profile versus S&P monthly returns
fig, axarr = plt.subplots(1, 2)
fmt = '%.0f%%'
xticks = mtick.FormatStrFormatter(fmt)

plot_data = pd.DataFrame(100.0 * vc.strat_data['index_fut_returns']['ES1'].rolling(21).sum())
plot_data['y'] = vc_output.combined_pnl_net['optim_weight'].rolling(21).sum()
axarr[0].scatter(plot_data['ES1'], plot_data['y'],
                 c=plot_data.index, cmap=cm.coolwarm)
axarr[0].set_title('Strategy returns versus S&P, monthly')
axarr[0].set_ylabel('PNL, multiple of max vega')
axarr[0].set_xlabel('S&P 500 return')
axarr[0].xaxis.set_major_formatter(xticks)

plot_data = pd.DataFrame(100.0 * vc.strat_data['index_fut_returns']['ES1'])
plot_data['y'] = vc_output['combined_pnl_net']['optim_weight']
axarr[1].scatter(plot_data['ES1'], plot_data['y'],
                 c=plot_data.index, cmap=cm.coolwarm)
axarr[1].set_title('Strategy returns versus S&P, daily')
axarr[1].set_ylabel('PNL, multiple of max vega')
axarr[1].set_xlabel('S&P 500 return')
axarr[1].xaxis.set_major_formatter(xticks)

# fig.colorbar(axarr[1].pcolor(plot_data.index))

# Scatterplot of vol vs S&P
fig, axarr = plt.subplots(1)
plot_data = pd.DataFrame(vc.strat_data['cm_vol_fut_returns'][21*5])
plot_data['y'] = pd.DataFrame(vc.strat_data['cm_vol_fut_returns'][21])
plot_data = plot_data[np.isfinite(plot_data).all(axis=1)]
plot_data = plot_data[(plot_data != 0.0).all(axis=1)]
axarr.scatter(plot_data[21*5], plot_data['y'],
                      c=plot_data.index, cmap=cm.coolwarm)
axarr.set_ylabel('1-month VIX future change, daily')
axarr.set_xlabel('5-month VIX future change, daily')
fmt = '%.0f%%'
axarr.set_ylim([-6, 6])
axarr.set_xlim([-3, 3])

'''
--------------------------------------------------------------------------------
Vega vs delta plots
--------------------------------------------------------------------------------
'''

vd = sm.strategies['equity_vs_vol']
vd_output = sm.outputs['equity_vs_vol']

# Risk and return
vd_output['combined_pnl'].rolling(21).sum().std()
vd_output['combined_pnl']['optim_weight'].rolling(21).sum().iloc[21:].quantile(0.005)
vd_output['combined_pnl_net'].rolling(21).sum().mean()
vd_output['combined_pnl_net'].rolling(21).sum().mean() \
    / vd_output['combined_pnl'].rolling(21).sum().std() * np.sqrt(12)

# Scatterplot of returns profile versus S&P monthly returns
fig, axarr = plt.subplots(1, 2)
fmt = '%.0f%%'
xticks = mtick.FormatStrFormatter(fmt)

plot_data = pd.DataFrame(100.0 * vd.strat_data['index_fut_returns']['ES1'].rolling(21).sum())
plot_data['y'] = vd_output['combined_pnl_net']['optim_weight'].rolling(21).sum()
axarr[0].scatter(plot_data['ES1'], plot_data['y'],
                 c=plot_data.index, cmap=cm.coolwarm)
axarr[0].set_title('Strategy returns versus S&P, monthly')
axarr[0].set_ylabel('PNL, multiple of max vega')
axarr[0].set_xlabel('S&P 500 return')
axarr[0].xaxis.set_major_formatter(xticks)

plot_data = pd.DataFrame(100.0 * vd.strat_data['index_fut_returns']['ES1'])
plot_data['y'] = vd_output['combined_pnl_net']['optim_weight']
axarr[1].scatter(plot_data['ES1'], plot_data['y'],
                 c=plot_data.index, cmap=cm.coolwarm)
axarr[1].set_title('Strategy returns versus S&P, daily')
axarr[1].set_ylabel('PNL, multiple of max vega')
axarr[1].set_xlabel('S&P 500 return')
axarr[1].xaxis.set_major_formatter(xticks)
im = axarr[0].imshow(np.random.random((10, 10)), vmin=0, vmax=1)
fig.colorbar(axarr[1].pcolor(plot_data.index))

# Robustness
plt.figure()
plt.plot(vd_output['combined_pnl_net']['optim_weight'].cumsum(), color='g')
plt.plot(vd_output['combined_pnl_net']['equal_weight'].cumsum(), color='k')
colormap = plt.get_cmap('coolwarm')
plt.gca().set_color_cycle([colormap(i)
    for i in np.linspace(0, 0.9, len(sim_perf_percentiles.columns))])
plt.plot(sim_perf_percentiles.cumsum())
plt.legend(['bayesian weight', 'equal weight',
            '1%', '10%', '25%', '50%', '75%', '90%', '99%'], loc=2)


# Basic plot
plt.figure()
plt.plot(vd_output['combined_pnl_net']['optim_weight'].cumsum())
plt.ylabel('cumulative strategy PNL, in vegas of max position size')
# plt.legend(['smart weighting', 'all signals equal weight'], loc=2)

# Positioning plot
plt.figure()
pos_signals = ['vol_spec_pos', 'index_spec_pos',
               'vol_spec_pos_chg_s', 'index_spec_pos_chg_s',
               'vol_spec_pos_chg_l', 'index_spec_pos_chg_l']
colormap = plt.get_cmap('coolwarm')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(pos_signals))])
plt.plot(vd_output['signal_pnl'][pos_signals].cumsum())
plt.legend(pos_signals, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# RV  plot
plt.figure()
rv_signals = ['rv_10', 'rv_21', 'rv_42', 'rv_63']
colormap = plt.get_cmap('spectral')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(rv_signals))])
plt.plot(vd_output['signal_pnl'][rv_signals].cumsum())
plt.legend(rv_signals, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# TS plot
plt.figure()
ts_signals = ['ts_0', 'ts_1', 'ts_5', 'ts_10']
colormap = plt.get_cmap('gist_heat')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(ts_signals))])
plt.plot(vd_output['signal_pnl'][ts_signals].cumsum())
plt.legend(ts_signals, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# Weights comparison plot
plt.figure()
plt.plot(vd_output['combined_pnl_net'].cumsum())
plt.ylabel('cumulative strategy PNL, in vegas of max position size')
plt.legend(['smart weighting', 'all signals equal weight'], loc=2)

# Signal robustness plot
plt.figure()
plt.plot(vd_output['combined_pnl_net'].cumsum())
plt.plot(sim_perf_percentiles.cumsum())
cols = ['optim_weight', 'equal_weight'] + [str(c)
    for c in sim_perf_percentiles.columns]
plt.legend(cols, loc=2)

# Scatterplot of vol vs S&P
fig, axarr = plt.subplots(1)
plot_data = pd.DataFrame(100.0 * vc.strat_data['index_fut_returns']['ES1'])
plot_data['y'] = pd.DataFrame(vc.strat_data['cm_vol_fut_returns'][21])
plot_data = plot_data[np.isfinite(plot_data).all(axis=1)]
plot_data = plot_data[(plot_data != 0.0).all(axis=1)]
axarr.scatter(plot_data['ES1'], plot_data['y'],
              c=plot_data.index, cmap=cm.coolwarm)
axarr.set_ylabel('1-month VIX future change, daily')
axarr.set_xlabel('S&P 500 return, daily')
fmt = '%.0f%%'
xticks = mtick.FormatStrFormatter(fmt)
axarr.xaxis.set_major_formatter(xticks)
axarr.set_ylim([-6, 6])
axarr.set_xlim([-10, 10])


'''
--------------------------------------------------------------------------------
Multistrat analysis
--------------------------------------------------------------------------------
'''

multistrat_pnl = pd.DataFrame(index=vd_output['combined_pnl_net'].index,
                              columns=['vd', 'vc', 'vrv'])
multistrat_pnl['vd'] = vd_output['combined_pnl_net']['optim_weight']
multistrat_pnl['vc'] = vc_output['combined_pnl_net']['optim_weight']
multistrat_pnl['vrv'] = optim_weight_pnl_net.diff(1)

for s in multistrat_pnl.columns:
    multistrat_pnl[s] = pd.to_numeric(multistrat_pnl[s])

multistrat_pnl.corr()
multistrat_pnl.rolling(window=21).sum().corr()

multistrat_pnl = multistrat_pnl[np.isfinite(multistrat_pnl).all(axis=1)]

multistrat_corr = multistrat_pnl.rolling(window=5).sum()\
    .ewm(com=126, min_periods=63).corr()
plt.figure()
plt.plot(multistrat_corr.major_xs('vd').transpose()[['vc', 'vrv']])
plt.plot(multistrat_corr.major_xs('vc').transpose()[['vrv']])
plt.ylabel('rolling correlation, exponentially weighted (1-year center of mass)')
plt.legend(['Equity-vs-Volatility & VIX Curve',
            'Equity-vs-Volatility & Vol RV',
            'VIX Curve & Vol RV'], loc=2)



# Illustrating positions
# Rows will be equity-vs-vol, columns are vix curve
# Equity-vs-vol "long" = short vol short stocks
# VIX curve "long" = long front short back
positions = pd.DataFrame(columns=['equity_vs_vol', 'vix_curve'])
positions['equity_vs_vol'] = pd.to_numeric(
    sm.outputs['equity_vs_vol']['positions']['optim_weight'])
positions['vix_curve'] = pd.to_numeric(
    sm.outputs['vix_curve']['positions']['optim_weight'])
positions = positions[np.isfinite(positions).all(axis=1)]

positions_grid_lo = [-np.inf, -0.5, -0.1, 0.1, 0.5]
positions_grid_hi = [-0.5, -0.1, 0.1, 0.5, np.inf]
bucket_names = ['large short', 'moderate short', 'neutral', 'moderate long', 'large long']
position_count = pd.DataFrame(index=bucket_names, columns=bucket_names)
for i in range(0, len(positions_grid_lo)):
    for j in range(0, len(positions_grid_lo)):
        ind = positions.index[(positions['equity_vs_vol'] < positions_grid_hi[i])
                            & (positions['equity_vs_vol'] >= positions_grid_lo[i])
                            & (positions['vix_curve'] < positions_grid_hi[j])
                            & (positions['vix_curve'] >= positions_grid_lo[j])]
        position_count.loc[bucket_names[i], bucket_names[j]] = \
            float(len(ind)) / len(positions)