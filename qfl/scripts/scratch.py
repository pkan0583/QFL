
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
import qfl.core.market_data as md
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
reload(utils)
reload(qfl_data)
reload(data_int)
reload(md)

from qfl.core.data_interfaces import QuandlApi, YahooApi, IBApi, FigiApi
from qfl.core.database_interface import DatabaseInterface as db, DatabaseUtilities as dbutils

db.initialize()

'''
--------------------------------------------------------------------------------
VIX futures
--------------------------------------------------------------------------------
'''

futures_series = 'FVS'
price_field = 'settle_price'
start_date = dt.datetime(2007, 1, 1)

etl.ingest_historical_futures_days_to_maturity(futures_series=futures_series,
                                               start_date=start_date,
                                               generic=True)
print('done!')

futures_prices = md.get_generic_futures_prices_from_series(
    futures_series=futures_series,
    start_date=start_date,
    include_contract_map=True)

start_dates = pd.to_datetime(
    futures_prices.index.get_level_values('date')).tolist()
end_dates = pd.to_datetime(futures_prices['maturity_date']).values.tolist()
futures_prices['days_to_maturity'] = utils.networkdays(
    start_date=start_dates,
    end_date=end_dates
)





'''
--------------------------------------------------------------------------------
Returns distributions
--------------------------------------------------------------------------------
'''

ticker = 'TLT'
price_field = 'last_price'
start_date = dt.datetime(2000, 1, 1)
return_window_days = 126

prices = md.get_stock_prices(ticker,
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

etl.load_historical_futures_prices(
    _db=db,
    dataset=dataset,
    futures_series=futures_series,
    start_date=dt.datetime(2012, 1, 1)
)

etl.ingest_historical_generic_futures_prices(
    _db=db,
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

import ib
from ib.opt import ibConnection
tws = ibConnection( host = 'localhost',port= 7496, clientId = 123)
tws.reqAccountUpdates(True,'accountnumber')

#############################################################################
# VIX futures
#############################################################################

reload(etl)
reload(qfl_data)
from qfl.core.data_interfaces import DatabaseInterface as db
from qfl.core.data_interfaces import QuandlApi
db.initialize()

# etl.load_historical_generic_vix_futures_prices(_db=db)

dataset = 'EUREX'
futures_series = 'FVS'
start_date = dt.datetime(2010, 1, 1)
contract_range = np.arange(1, 9)
_db = db

data = qfl_data.QuandlApi.retrieve_historical_vstoxx_futures_prices(
    start_date=start_date
)

etl.load_historical_futures_prices(
    _db=db,
    start_date=start_date,
    futures_series=futures_series,
    dataset=dataset
)

etl.ingest_historical_generic_futures_prices(
    _db=db,
    start_date=start_date,
    futures_series='FVS',
    source_series='EUREX_FVS',
    contract_range=contract_range,
    dataset='CHRIS'
)

# Get updated VIX futures data
vix_data = qfl_data.QuandlApi.update_daily_futures_prices(
    date=dt.datetime.today(),
    dataset='CBOE',
    futures_series='VX',
    contract_range=np.arange(0, 10)
)

v2x_data = qfl_data.QuandlApi.update_daily_futures_prices(
    date=dt.datetime.today(),
    dataset='EUREX',
    futures_series='FVS',
    contract_range=np.arange(0, 10)
)

etl.update_futures_prices(_db=db,
                          dataset='EUREX',
                          futures_series='FVS',
                          contract_range=np.arange(0,10))

futures_data = v2x_data
futures_series = 'FVS'
_db = db

futures_data['open_interest'] = \
    futures_data['Prev. Day Open Interest']

# Map to appropriate contact
series_data = _db.get_futures_series(futures_series=futures_series)
series_id = series_data['id'].values[0]

where_str = " series_id = " + str(series_id) \
            + " and maturity_date >= '" + dt.datetime.today().date().__str__() + "'"

futures_contracts_data = _db.get_data(
    table_name='futures_contracts',
    where_str=where_str)

futures_data = futures_data.reset_index()

df = pd.merge(left=futures_contracts_data[['series_id', 'ticker']],
              right=futures_data,
              on='ticker')
cols = ['series_id', 'date', 'close_price', 'high_price', 'low_price',
        'open_price', 'settle_price', 'volume', 'open_interest']
cols = list(set(cols).intersection(set(df.columns)))

df = df[cols]




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



# note codes for VIX futures positioning
# CFTC/TIFF_CBOE_VX_ALL

#############################################################################
# Database stuff
#############################################################################


reload(etl)
reload(qfl_data)
from qfl.core.database_interface import DatabaseInterface as db

# Database stuff
db.initialize()




etl.daily_equity_price_ingest()

date = utils.closest_business_day()
data_source = 'yahoo'

etl.ingest_historical_equity_prices(ids=None,
                                    start_date=etl.default_start_date,
                                    end_date=date,
                                    data_source=data_source,
                                    _db=db)

equities, equity_prices_table, ids, equity_tickers, rows = \
    etl._prep_equity_price_ingest(ids=None, _db=db)


etl.update_equity_prices(ids=None, data_source_name='yahoo', _db=db)
print('done!')



_id = ids[0]
etl._ingest_historical_equity_prices(_id,
                                     start_date=date,
                                     end_date=date,
                                     data_source_name='yahoo',
                                     _db=db)



id_ = 3642
etl.update_option_prices_one(id_=id_, db=db)

ids = [3642, 3643]
etl.update_option_prices(ids, 'yahoo', db)

etl._ingest_historical_equity_prices(_id=3642,
                                     start_date=None,
                                     end_date=None,
                                     data_source_name='yahoo',
                                     _db=db)

# Test load of options prices from yahoo
start_date = dt.datetime(1990, 1, 1)
end_date = dt.datetime.today()

options_table = db.get_table(table_name='equity_options')

tmp = YahooApi.retrieve_options_data('ABBV')
raw_data = tmp[0]
unique_symbols = np.unique(raw_data.index.get_level_values(level='Symbol'))
unique_symbols = [str(symbol) for symbol in unique_symbols]

# Load attributes
option_attributes = pd.DataFrame(raw_data.index.get_level_values(level='Symbol'))
option_attributes = option_attributes.rename(columns={'Symbol': 'ticker'})
option_attributes['option_type'] = raw_data.index.get_level_values(
    level='Type')
option_attributes['strike_price'] = raw_data.index.get_level_values(
    level='Strike')
option_attributes['maturity_date'] = raw_data.index.get_level_values(
    level='Expiry')
option_attributes['underlying_id'] = db.get_equity_ids(
    equity_tickers=raw_data['Underlying'])

db.execute_db_save(df=option_attributes,
                   table=options_table,
                   use_column_as_key='ticker')


# Get their ID's
t = tuple([str(ticker) for ticker in option_attributes['ticker']])
q = 'select ticker, id from equity_options where ticker in {0}'.format(t)
ticker_id_map = db.read_sql(query=q)
ticker_id_map.index = ticker_id_map['ticker']

option_prices_table = db.get_table(table_name='equity_option_prices')

# Load prices
option_prices = pd.DataFrame(columns=['date'])
option_prices['date'] = raw_data['Quote_Time'].dt.date
option_prices['quote_time'] = raw_data['Quote_Time'].dt.time
option_prices['last_price'] = raw_data['Last']
option_prices['bid_price'] = raw_data['Bid']
option_prices['ask_price'] = raw_data['Ask']
option_prices['iv'] = raw_data['IV']
option_prices['volume'] = raw_data['Vol']
option_prices['open_interest'] = raw_data['Open_Int']
option_prices['spot_price'] = raw_data['Underlying_Price']
option_prices['iv'] = option_prices['iv'].str.replace('%', '')
option_prices['iv'] = option_prices['iv'].astype(float) / 100.0

ids = ticker_id_map.loc[option_attributes['ticker'], 'id']
option_prices = option_prices.reset_index()
option_prices['id'] = ids.values

db.execute_db_save(df=option_prices,
                   table=option_prices_table)







# Testing archive of historical prices

equities = db.get_data(table_name='equities', index_table=True)
ids = equities.index.tolist()

rows = equities.loc[ids].reset_index()
equity_tickers = rows['ticker'].tolist()
equity_tickers = [str(ticker) for ticker in equity_tickers]

equity_tickers = equity_tickers[0:1]

ids = db.get_equity_ids(equity_tickers)

date = qfl.utilities.basic_utilities.closest_business_day()

expiry_dates, links = pdata.YahooOptions('ABBV') \
    ._get_expiry_dates_and_links()

data = pdata.YahooOptions('ABBV').get_near_stock_price(
    above_below=20, expiry=expiry_dates[0])

def hello(c):
    c.drawString(100, 100, "Hello World")

# Create a canvas
c = canvas.Canvas(filename="hello.pdf",
                  pagesize=letter,
                  bottomup=1,
                  pageCompression=0,
                  verbosity=0,
                  encrypt=None)
width, height = letter

# Draw some stuff


hello(c)
c.showPage()
c.save()


# FIGI identifiers - cool!
api_key = "471197f1-50fe-429b-9e11-a6828980e213"
req_data = [{"idType":"TICKER","idValue":"YHOO","exchCode":"US"}]
r = requests.post('https://api.openfigi.com/v1/mapping',
                  headers={"Content-Type": "text/json",
                           "X-OPENFIGI-APIKEY": api_key},
                  json=req_data)




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