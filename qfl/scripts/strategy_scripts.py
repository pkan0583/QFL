# execfile("startup.py")

import datetime as dt
import numpy as np
import pandas as pd
import struct
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sn
from sklearn.decomposition import FactorAnalysis, PCA

import qfl.core.calcs as calcs
import qfl.core.market_data as md
import qfl.core.constants as constants
import qfl.macro.macro_models as macro
import qfl.utilities.basic_utilities as utils
import qfl.core.portfolio_utils as putils

from qfl.core.database_interface import DatabaseInterface as db
from qfl.utilities.chart_utilities import format_plot
from qfl.utilities.nlp import DocumentsAnalyzer as da
import qfl.utilities.statistics as stats
from qfl.core.market_data import VolatilitySurfaceManager
from qfl.macro.macro_models import FundamentalMacroModel

import qfl.models.volatility_factor_model as vfm
from qfl.utilities.statistics import RollingFactorModel
from qfl.models.volatility_factor_model import VolatilityFactorModel

import qfl.strategies.strategies as strat
from qfl.strategies.vol_fut_curve import VixCurveStrategy
from qfl.strategies.vega_vs_delta import VegaVsDeltaStrategy
from qfl.strategies.volswap_rv import VolswapRvStrategy
from qfl.strategies.strategies import PortfolioOptimizer

import logging

db.initialize()
logger = logging.getLogger()

'''
--------------------------------------------------------------------------------
New centralized approach
--------------------------------------------------------------------------------
'''

import qfl.strategies.strategy_master as s_m
reload(s_m)
from qfl.strategies.strategy_master import StrategyMaster

sm = StrategyMaster()

sm.initialize_vix_curve()
sm.initialize_equity_vs_vol()

sm.initialize_macro_model()
sm.initialize_volatility_factor_model()
sm.initialize_volatility_rv()

# Current VIX term structure
cols = ['days_to_maturity', 'settle_price']
vix_spot = md.get_generic_index_prices(tickers=['VIX'])
vix_ts = md.get_futures_prices_by_series(futures_series='VX')[cols]
vix_ts.loc[(md.market_data_date, 'VIX'), ['days_to_maturity', 'settle_price']] = [0, 13.29]
vix_ts = vix_ts.sort_values('days_to_maturity')

# Latest signals
sm.outputs['vix_curve'].signal_data.loc[md.market_data_date]
sm.outputs['vix_curve'].signal_data_z.loc[md.market_data_date]

vd = VegaVsDeltaStrategy()
vd.initialize_data()


from qfl.strategies.vega_vs_delta import Person, Dude, Chick

bob = Dude("bob")
jen = Chick("jen")



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


import qfl.strategies.volswap_rv as vrv
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


###############################################################################
# Optimization
###############################################################################

included_signals = rv_iv_signals.columns \
                   + rv_signals.columns \
                   + ts_signals.columns \
                   + macro_signals.columns

signal_er = daily_pnl.mean(axis=0)  \
            * constants.trading_days_per_year

# Single in-sample covariance matrix
signal_cov = daily_pnl.cov() \
             * constants.trading_days_per_year
signal_corr = daily_pnl.corr()

optim_output = vrv.compute_signal_portfolio_optimization(
    signal_er=signal_er,
    signal_corr=signal_corr,
    signal_cov=signal_cov,
    signal_er_corr_shrinkage=0.80,
    signal_se_beta_to_vol=0.75)

optim_weight_pnl_gross = pd.DataFrame(index=pnl.index,
                                      columns=['composite'])
optim_weight_pnl_gross['composite'] = 0
for sig in included_signals:
    optim_weight_pnl_gross['composite'] += pnl[sig] \
        * optim_output.weights.loc[sig, 'weight']

# CLEANING: MAKE SURE YOU CHECK THIS


# Transaction costs: every day we amortize gross vega / TC (improve this)
tc_vegas = 0.125
dates = portfolio_summary_df.unstack('index').index
tc_ts = pd.DataFrame(index=dates, columns=['tc'])
tc_ts['tc'] = tc_vegas * portfolio_summary_df['gross_vega'].mean() / 63.0
tc_ts['cum_tc'] = tc_ts['tc'].cumsum()

# Unweighted-average signal PNL
included_signals_unw = rv_iv_signals.columns[0:3] \
                   + rv_signals.columns[0:3] \
                   + ts_signals.columns

# CAREFUL: OUTPUT CLEANING, NOT SURE IF THIS IS CORRECT
equal_weight_pnl_gross = pnl[included_signals].fillna(method='ffill').mean(axis=1)
tmp = equal_weight_pnl_gross.diff(1)
tmp = tmp[np.abs(tmp) < 100.0]
equal_weight_pnl_gross = tmp.cumsum()

equal_weight_pnl_net = equal_weight_pnl_gross - tc_ts['cum_tc']
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
# Pure quantiles analysis with daily pnl
###############################################################################

rv_iv_signals = vrv.initialize_rv_iv_signals()
iv_signals = vrv.initialize_iv_signals()
ts_signals = vrv.initialize_ts_signals()

rv_iv_pnl, rv_iv_pos, rv_iv_pctile \
    = vrv.compute_signal_quantile_performance(signal_data=rv_iv_signals)

iv_pnl, iv_pos, iv_pctile \
    = vrv.compute_signal_quantile_performance(signal_data=iv_signals)

ts_pnl, ts_pos, ts_pctile \
    = vrv.compute_signal_quantile_performance(signal_data=ts_signals)

rv_pnl, rv_pos, rv_pctile \
    = vrv.compute_signal_quantile_performance(signal_data=rv_signals)

# signal_pnl, signal_pos, signal_pctile = vrv.compute_master_backtest()
print('done!')

# Save
vrv_data = vrv.strat_data
vrv_settings = vrv.settings
vrv_calc = vrv.calc

# Restore
vrv.strat_data = vrv_data
vrv.settings = vrv_settings
vrv.calc = vrv_calc
vrv.set_universe()

# Analyze results

signal_pnl = rv_pnl
signal_positions = rv_pos
quantiles = signal_pnl.keys()
signal_com = np.unique(rv_pnl[1.0].columns.get_level_values(None))

buy_q = 0.80
sell_q = 0.20
tc_vega = 0

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


###############################################################################
# playing with the idea of an option strategy
###############################################################################

reload(md)
from qfl.core.market_data import VolatilitySurfaceManager
vsm = VolatilitySurfaceManager()
vsm.load_data(tickers=md.get_etf_vol_universe(), start_date=dt.datetime(2010, 1, 1))

trade_frequency_days = 5
trade_tenor_month = 3
quantiles = [0.10, 0.30, 0.70, 0.90, 1.0]
buy_q = 0.90
sell_q = 0.30
hedge_ratio_vov_weight = 0.50

call_deltas_initial_position = [0.40, 0.60]
relative_sizes_initial_position = [1.0, 1.0]
vrv_data = vrv.strat_data.vrv_data

signal_name = 'rv_iv_21'
signal_data = pd.DataFrame(rv_iv_signals[signal_name])

sec_cols = ['instrument',
            'underlying',
            'trade_date',
            'maturity_date',
            'strike',
            'option_type',
            'quantity']

# Template data
start_date = vrv.settings.start_date
sample_data = vsm.get_data(tickers=['SPY'],
                           start_date=start_date)
dates = sample_data.index.get_level_values('date').sort_values()

maturity_dates = vsm.get_roll_schedule(tickers=['SPY'],
                                       start_date=vrv.settings.start_date)['SPY']

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
VIX curve
--------------------------------------------------------------------------------
'''

vc = VixCurveStrategy()

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

# Save this guy down because at least for now he gets overriden when i run VD
vc_pnl_net = vc_output.combined_pnl_net.copy(deep=True)

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
        signal_data=vc_output.signal_output.signals_data,
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

# Impact of smart weighting
plt.figure()
plt.plot(vc_output.combined_pnl_net.cumsum())
plt.ylabel('cumulative strategy PNL, in vegas of max position size')
plt.legend(['bayesian weighting', 'naive equal weight'], loc=2)

# Signal robustness plot
plt.figure()
plt.plot(vc_output.combined_pnl_net['optim_weight'].cumsum(), color='k')
plt.plot(vc_output.combined_pnl_net['equal_weight'].cumsum(), color='g')
colormap = plt.get_cmap('coolwarm')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(sim_perf_percentiles.columns))])
plt.plot(sim_perf_percentiles.cumsum())
plt.plot(sim_perf_percentiles.cumsum())
cols = ['optim_weight', 'equal_weight'] + [str(c) for c in sim_perf_percentiles.columns]
plt.legend(cols, loc=2)
plt.ylabel('cumulative strategy PNL, in vegas of max position size')

# Scatterplot of returns profile versus S&P monthly returns
fig, axarr = plt.subplots(1, 2)
fmt = '%.0f%%'
xticks = mtick.FormatStrFormatter(fmt)

plot_data = pd.DataFrame(100.0 * vc.strat_data.index_fut_returns['ES1'].rolling(21).sum())
plot_data['y'] = vc_output.combined_pnl_net['optim_weight'].rolling(21).sum()
axarr[0].scatter(plot_data['ES1'], plot_data['y'],
                 c=plot_data.index, cmap=cm.coolwarm)
axarr[0].set_title('Strategy returns versus S&P, monthly')
axarr[0].set_ylabel('PNL, multiple of max vega')
axarr[0].set_xlabel('S&P 500 return')
axarr[0].xaxis.set_major_formatter(xticks)

plot_data = pd.DataFrame(100.0 * vc.strat_data.index_fut_returns['ES1'])
plot_data['y'] = vc_output.combined_pnl_net['optim_weight']
axarr[1].scatter(plot_data['ES1'], plot_data['y'],
                 c=plot_data.index, cmap=cm.coolwarm)
axarr[1].set_title('Strategy returns versus S&P, daily')
axarr[1].set_ylabel('PNL, multiple of max vega')
axarr[1].set_xlabel('S&P 500 return')
axarr[1].xaxis.set_major_formatter(xticks)

# fig.colorbar(axarr[1].pcolor(plot_data.index))

# Scatterplot of vol vs S&P
fig, axarr = plt.subplots(1)
plot_data = pd.DataFrame(vc.strat_data.cm_vol_fut_returns[21*5])
plot_data['y'] = pd.DataFrame(vc.strat_data.cm_vol_fut_returns[21])
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
Vega vs delta
--------------------------------------------------------------------------------
'''

vd = VegaVsDeltaStrategy()

# Beginning of semi-clean VIX futures data
start_date = dt.datetime(2007, 3, 26)
change_window_days_short = 5
change_window_days_long = 252
holding_period_days = 1
signals_z_cap = 1.0
vol_target_com = 63
rolling_beta_com = 63
tc = 0.05

corr_r_shrinkage = 0.80
corr_er_shrinkage = 0.60
er_se_beta_to_er = 0.25
er_se_beta_to_vol = 1.0

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

vd_pnl_net = vd_output.combined_pnl_net.copy(deep=True)

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


# Assessing time varying expected return
# er_com = 512
# er_outlier_com = 512
# cap_factor = 2.5
#
# signal_perf = vd_output.combined_pnl_net
# signal_perf_bracket = signal_perf.ewm(com=er_outlier_com).std() * cap_factor
# signal_perf_capped = np.minimum(signal_perf_bracket,
#                                 np.maximum(signal_perf, -signal_perf_bracket))
# signal_er = signal_perf_capped.iloc[252:].ewm(com=er_com).mean()
# plt.plot(signal_er)


# Scatterplot of returns profile versus S&P monthly returns
fig, axarr = plt.subplots(1, 2)
fmt = '%.0f%%'
xticks = mtick.FormatStrFormatter(fmt)

plot_data = pd.DataFrame(100.0 * vd.strat_data.index_fut_returns['ES1'].rolling(21).sum())
plot_data['y'] = vd_output.combined_pnl_net['optim_weight'].rolling(21).sum()
axarr[0].scatter(plot_data['ES1'], plot_data['y'],
                 c=plot_data.index, cmap=cm.coolwarm)
axarr[0].set_title('Strategy returns versus S&P, monthly')
axarr[0].set_ylabel('PNL, multiple of max vega')
axarr[0].set_xlabel('S&P 500 return')
axarr[0].xaxis.set_major_formatter(xticks)

plot_data = pd.DataFrame(100.0 * vd.strat_data.index_fut_returns['ES1'])
plot_data['y'] = vd_output.combined_pnl_net['optim_weight']
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
plt.plot(vd_output.combined_pnl_net['optim_weight'].cumsum(), color='g')
plt.plot(vd_output.combined_pnl_net['equal_weight'].cumsum(), color='k')
colormap = plt.get_cmap('coolwarm')
plt.gca().set_color_cycle([colormap(i)
    for i in np.linspace(0, 0.9, len(sim_perf_percentiles.columns))])
plt.plot(sim_perf_percentiles.cumsum())
plt.legend(['bayesian weight', 'equal weight',
            '1%', '10%', '25%', '50%', '75%', '90%', '99%'], loc=2)


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
colormap = plt.get_cmap('coolwarm')
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, len(pos_signals))])
plt.plot(vd_output.signal_output.signal_pnl[pos_signals].cumsum())
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

# Scatterplot of vol vs S&P
fig, axarr = plt.subplots(1)
plot_data = pd.DataFrame(100.0 * vc.strat_data.index_fut_returns['ES1'])
plot_data['y'] = pd.DataFrame(vc.strat_data.cm_vol_fut_returns[21])
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

multistrat_pnl = pd.DataFrame(index=vd_output.combined_pnl_net.index,
                              columns=['vd', 'vc', 'vrv'])
multistrat_pnl['vd'] = vd_pnl_net['optim_weight']
multistrat_pnl['vc'] = vc_pnl_net['optim_weight']
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