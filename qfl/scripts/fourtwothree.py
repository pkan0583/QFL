import pandas as pd
import pandas_datareader.data as pdata
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sn
import datetime as dt

import qfl.core.market_data as md
import qfl.utilities.chart_utilities as chart_utils
import qfl.macro.macro_models as macro


"""
-------------------------------------------------------------------------------
Hedge fund AUM
-------------------------------------------------------------------------------
"""

text_color_scheme = 'r'
hf_aum = pd.read_excel(io='data/AUM_HF.xls', sheetname='raw')

hf_aum.index = hf_aum['YEAR']
fig = plt.figure(figsize=[12, 6])
ax = fig.add_subplot(1,1,1)
ax.plot(hf_aum['AUM'])
ax.set_ylabel('Hedge fund industry AUM, $ billions')
ax.set_xlim([hf_aum['YEAR'].min(), hf_aum['YEAR'].max()])
plt.xticks(hf_aum['YEAR'].values)
for xy in zip(hf_aum['YEAR'].values, hf_aum['AUM'].values):
    ax.annotate('%s' % int(xy[1]),
                xy=xy,
                textcoords='data',
                color=text_color_scheme)
plt.savefig('figures/aum_fig.png',
            facecolor='k',
            edgecolor='k',
            transparent=True)

"""
-------------------------------------------------------------------------------
GURU chart
-------------------------------------------------------------------------------
"""

guru_price = pd.read_excel('data/guru.xlsx')
guru_start_date = guru_price['date'].min()
guru_price.index = guru_price['date']

spx_price = md.get_equity_index_prices(['SPX'], start_date=guru_start_date)
spx_price = spx_price['last_price']

prices = pd.DataFrame(data=guru_price)
prices['SPX'] = spx_price.reset_index(level='ticker', drop=True)
prices = prices.rename(columns={'price': 'GURU'})

prices = prices.sort_index()
prices = prices[['GURU', 'SPX']]

daily_returns = prices / prices.shift(1) - 1
cumulative_returns = prices / prices.iloc[0] - 1
vols = daily_returns.std()

cumulative_returns['diff'] = cumulative_returns['GURU'] \
                           - cumulative_returns['SPX']

cumulative_returns['risk-adj diff'] = cumulative_returns['GURU'] \
    - vols['GURU'] / vols['SPX'] * cumulative_returns['SPX']

# cols = ['diff', 'risk-adj diff']
# legend_text = ['GURU versus S&P', 'GURU versus S&P, risk-adjusted']
fig = plt.figure(figsize=[12, 6])
ax = fig.add_subplot(1,1,1)
ax.plot(100 * cumulative_returns['risk-adj diff'], 'b')
ax.set_ylabel('Cumulative outperformance of GURU vs S&P 500, %')
yticks = mtick.FormatStrFormatter('%.0f%%')
ax.yaxis.set_major_formatter(yticks)

plt.savefig('figures/guru.png',
            facecolor='k',
            edgecolor='k',
            transparent=True)


"""
-------------------------------------------------------------------------------
EUR positioning
-------------------------------------------------------------------------------
"""

import quandl as ql
sn.set_style('white')

spx_positioning = ql.get('CFTC/TIFF_CME_ES_ALL')
eur_positioning = ql.get('CFTC/TIFF_CME_EC_ALL')

spx_positioning['lev_net'] = spx_positioning['Lev Money Long Positions'] \
                           - spx_positioning['Lev Money Short Positions']
eur_positioning['lev_net'] = eur_positioning['Lev Money Long Positions'] \
                           - eur_positioning['Lev Money Short Positions']

eurusd = ql.get("FRED/DEXUSEU")
usdjpy = ql.get("FRED/AEXJPUS")

eur_data = pd.DataFrame(columns=['spot'], data=eurusd)
eur_data['spot'] = eurusd
eur_data['lev_net'] = eur_positioning['lev_net']
eur_data['lev_net'] = eur_data['lev_net'].fillna(method='ffill')
eur_data['lev_pnl'] = eur_data['lev_net'].shift(1) * eur_data['spot'].diff(1)
eur_data['daily_return'] = eur_data['spot'] / eur_data['spot'].shift(1) - 1
eur_data['volatility'] = np.sqrt(252.0) * eur_data['daily_return']\
    .rolling(window=21, center=False).std()

plot_start_date = dt.datetime(2015, 2, 1)
plot_data = eur_data[eur_data.index >= plot_start_date]

fig, ax1 = plt.subplots()
ax1.plot(plot_data['volatility'], 'b-')
ax1.set_ylabel('eurusd volatility', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(plot_data['lev_net'], 'k')
ax2.set_ylabel('eur positioning', color='k')
for tl in ax2.get_yticklabels():
    tl.set_color('k')

plt.show()

plt.figure()
plt.plot(plot_data['lev_pnl'].cumsum())


"""
-------------------------------------------------------------------------------
Volatility convexity chart
-------------------------------------------------------------------------------
"""

# Var/vol with a cap

var_strike = 18
vol_strike = 16
cap_factor = 2.5
cap_sale_ratio = 0.6
vol_grid = np.arange(0.0, 100.0, 5.0)

df = pd.DataFrame(index=vol_grid, columns=['var', 'vol', 'cap'])
df['var'] = (vol_grid ** 2 - var_strike ** 2) / (2 * var_strike)
df['vol'] = vol_grid - vol_strike
df['cap'] = (vol_grid ** 2 - (var_strike * cap_factor) ** 2) / (2 * var_strike)
df['cap'][df['cap'] < 0] = 0
df['cap_ratio'] = df['cap'] * cap_sale_ratio
df['var_vol'] = df['var'] - df['vol']
df['package'] = df['var_vol'] - df['cap_ratio']

plt.figure()
plt.plot(df['var_vol'])
plt.plot(df['cap_ratio'])
plt.plot(df['package'])


"""
-------------------------------------------------------------------------------
Skew chart
-------------------------------------------------------------------------------
"""

# Skew trade

spot_px = 1.06
call_strike = 1.07
put_strike = 1.05
tenor_in_days = 5
risk_free = 0.0
call_vol = 0.15
put_vol = 0.16
event_vol = 0.05

call_px = calcs.black_scholes_price(spot=spot_px,
                                    strike=call_strike,
                                    tenor_in_days=tenor_in_days,
                                    risk_free=risk_free,
                                    ivol=call_vol,
                                    div_amt=0,
                                    option_type='c')

put_px = calcs.black_scholes_price(spot=spot_px,
                                   strike=put_strike,
                                   tenor_in_days=tenor_in_days,
                                   risk_free=risk_free,
                                   ivol=put_vol,
                                   div_amt=0,
                                   option_type='p')

hedge_ratio = call_px / put_px

call_delta = calcs.black_scholes_delta(spot=spot_px,
                                       strike=call_strike,
                                       tenor_in_days=tenor_in_days,
                                       risk_free=risk_free,
                                       ivol=call_vol-event_vol,
                                       div_amt=0,
                                       option_type='c')

put_delta = calcs.black_scholes_delta(spot=spot_px,
                                      strike=put_strike,
                                      tenor_in_days=tenor_in_days,
                                      risk_free=risk_free,
                                      ivol=put_vol-event_vol,
                                      div_amt=0,
                                      option_type='p')

spot_grid = np.arange(1.0, 1.12, 0.005)
cols = ['call', 'put', 'delta_hedge', 'risk_reversal']
scenario_analysis = pd.DataFrame(index=spot_grid, columns=cols)

scenario_analysis['call'] = calcs.black_scholes_price(spot=spot_grid,
                                                      strike=call_strike,
                                                      tenor_in_days=tenor_in_days-1,
                                                      risk_free=risk_free,
                                                      ivol=call_vol - event_vol,
                                                      div_amt=0,
                                                      option_type='c')

scenario_analysis['put'] = calcs.black_scholes_price(spot=spot_grid,
                                                     strike=put_strike,
                                                     tenor_in_days=tenor_in_days-1,
                                                     risk_free=risk_free,
                                                     ivol=put_vol - event_vol,
                                                     div_amt=0,
                                                     option_type='p')

scenario_analysis['delta_hedge'] = -(call_delta - put_delta * hedge_ratio) \
                                   * (spot_grid - 1.06)

scenario_analysis['risk_reversal'] = scenario_analysis['call'] \
                                     - scenario_analysis['put'] * hedge_ratio \
                                     + scenario_analysis['delta_hedge'] * 0.925

fig, ax = plt.subplots()
ax.plot(scenario_analysis['risk_reversal'] * 100.0)
fmt = '%.2f%%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)
plt.ylabel('PNL, % of notional')
plt.xlabel('EURUSD after the event')
ax.set_xlim([1.0, 1.12])


"""
-------------------------------------------------------------------------------
Macro model
-------------------------------------------------------------------------------
"""


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