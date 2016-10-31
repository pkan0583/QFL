import pandas as pd
import datetime as dt
import numpy as np
from scipy.stats import norm, expon, gamma, invgamma, beta, nct
from scipy.stats import t, normaltest
from scipy.stats import multivariate_normal, wishart, invwishart, dirichlet
from scipy.stats import bernoulli
from scipy import interpolate
import pymc
import matplotlib.pyplot as plt
import qfl.utilities.basic_utilities as utils
import qfl.core.constants as constants
from qfl.utilities.bayesian_modeling import BayesianTimeSeriesDataCleaner

def windsorize(df=None, z=2.0):

    df = df.copy(deep=True)
    df_sd = df.std()
    if isinstance(df, pd.DataFrame):
        for col in df.columns:

            ind = df.index[df[col] > z * df_sd[col]]
            df.loc[ind, col] = z * df_sd[col]

            ind = df.index[df[col] < -z * df_sd[col]]
            df.loc[ind, col] = -z * df_sd[col]
    elif isinstance(df, pd.Series):
        df[df > z * df_sd] = z * df_sd
        df[df < -z * df_sd] = -z * df_sd
    return df


def compute_uncertainty_adj_portfolio(**kwargs):

    er = kwargs.get('expected_return_mean', None)
    volr = kwargs.get('return_covariance', None)
    er_sd = kwargs.get('expected_return_covariance', None)


def compute_seasonality_adj_vol_futures_prices(futures_data=None,
                                               futures_contracts=None,
                                               series_name='VX',
                                               price_field='settle_price',
                                               vix_spot_tenor=30,
                                               base_trading_days=23,
                                               trading_days_adj_factor=0.5,
                                               december_effect_vol_points=0.25):

    """
    :param futures_data: DataFrame indexed on [date, ticker] with columns
        [contract_ticker, maturity_date, days_to_maturity, <price fields>]
    :param futures_contracts: DataFrame with columns [ticker, maturity_date]
    :param series_name:
    :param price_field: String, should be a column in futures_data
    :param vix_spot_tenor: calendar days corresponding to the underlying option
        portfolio of the volatility index. For VIX this is 30
    :param base_trading_days: the # of business days to normalize to
    :param trading_days_adj_factor: what % of the theoretical day count
        adjustment to apply (1 = full adjustment)
    :param december_effect_vol_points: additional adjustment for December
    :return:
    """

    # Ensure appropriate formatting
    futures_contracts['maturity_date'] \
        = pd.to_datetime(futures_contracts['maturity_date'])
    futures_data['maturity_date'] \
        = pd.to_datetime(futures_data['maturity_date'])

    # Get appropriate calendar (TODO: replace)
    calendar_name = 'UnitedStates'
    if series_name == 'FVS':
        calendar_name = 'EuropeanCentralBank'

    futures_contracts['option_maturity_date'] = futures_contracts[
        'maturity_date'] + dt.timedelta(days=vix_spot_tenor)

    futures_contracts['option_trading_days'] = utils.networkdays(
        start_date=futures_contracts['maturity_date'].values,
        end_date=futures_contracts['option_maturity_date'].values,
        calendar_name=calendar_name
    )

    futures_contracts = futures_contracts.rename(
        columns={'ticker': 'contract_ticker'})
    futures_data = pd.merge(
        left=futures_data.reset_index(),
        right=futures_contracts[['contract_ticker', 'option_trading_days']],
        on='contract_ticker')
    futures_data.index = [futures_data.date, futures_data.ticker]

    futures_data['seasonality_adj_price'] = futures_data[price_field] * \
         np.sqrt((base_trading_days / futures_data['option_trading_days'])
         ** trading_days_adj_factor)

    futures_data['december'] = futures_data['maturity_date'].dt.month == 12
    ind = futures_data.index[futures_data['december'] == True]
    if len(ind) > 0:
        futures_data.loc[ind, 'seasonality_adj_price'] \
            += december_effect_vol_points
    return futures_data


def compute_rolling_futures_returns(generic_futures_data=None,
                                    price_field='settle_price',
                                    level_change=False,
                                    days_to_zero_around_roll=1):

    cols = generic_futures_data[price_field].columns
    ind = generic_futures_data.index[
        generic_futures_data['days_to_maturity'][cols[0]] >
        generic_futures_data['days_to_maturity'][cols[0]].shift(
            days_to_zero_around_roll)]
    if level_change:
        futures_returns = generic_futures_data[price_field] - \
                          generic_futures_data[price_field].shift(1)
    else:
        futures_returns = generic_futures_data[price_field] / \
                          generic_futures_data[price_field].shift(1) - 1
    futures_returns.loc[ind] = 0
    return futures_returns


def compute_constant_maturity_futures_prices(generic_futures_data=None,
                                             constant_maturities_in_days=None,
                                             price_field='settle_price',
                                             spot_prices=None,
                                             volatilities=False):

    """
    Calculate constant maturity futures prices from generic futures prices.
    :param generic_futures_data: DataFrame indexed on date and ticker,
    with columns [<price field>, days_to_maturity]
    :param constant_maturities_in_days: list of integers
    :param price_field: the price field to use from generic_futures_data
    :param spot_prices: if using a spot price for the term structure,
    this is a DataFrame indexed on dates
    :return: Pandas DataFrame with columns = constant_maturities_in_days,
    and values equal to constant-maturity prices
    """

    generic_futures_prices = generic_futures_data[price_field].unstack('ticker')
    futures_ttm = generic_futures_data['days_to_maturity'].unstack('ticker')

    for col in generic_futures_prices.columns:
        generic_futures_prices[col] = pd.to_numeric(generic_futures_prices[col])

    for col in futures_ttm.columns:
        futures_ttm[col] = pd.to_numeric(futures_ttm[col])

    if spot_prices is not None:

        orig_cols = generic_futures_prices.columns.tolist()

        generic_futures_prices['spot'] = spot_prices
        generic_futures_prices = generic_futures_prices[['spot'] + orig_cols]

        futures_ttm['spot'] = 0
        futures_ttm = futures_ttm[['spot'] + orig_cols]

    constant_maturity_futures_prices = pd.DataFrame(
        index=generic_futures_prices.index,
        columns=constant_maturities_in_days)
    dates = generic_futures_prices.index.get_level_values('date')

    for t in range(0, len(generic_futures_prices)):
        date = dates[t]

        ind = (np.isfinite(futures_ttm.loc[date])) \
                  & (np.isfinite(generic_futures_prices.loc[date])
                  & futures_ttm.loc[date] >= 0.0)

        interp_values = generic_futures_prices.loc[date][ind]

        if volatilities:
            interp_values **= 2.0

        try:
            f = interpolate.interp1d(x=futures_ttm.loc[date][ind],
                                     y=interp_values,
                                     kind='linear',
                                     fill_value='extrapolate',
                                     bounds_error=False)

            constant_maturity_futures_prices.loc[date] \
                = f(constant_maturities_in_days)
        except:
            x=1

    constant_maturity_futures_prices[
        constant_maturity_futures_prices < 0] = np.nan

    if volatilities:
        constant_maturity_futures_prices **= 0.5

    return constant_maturity_futures_prices


def clean_implied_vol_data(tickers=None,
                           stock_prices=None,
                           ivol=None,
                           ref_ivol_ticker=None,
                           calendar_name='UnitedStates'):

    orig_ivol = ivol.copy(deep=True)
    ivol = ivol.copy(deep=True)
    # stock_returns = stock_prices.diff(1) / stock_prices.shift(1)

    # Data should come in unstacked (tickers in columns)
    if len(ivol.index.names) > 1:
        ivol = ivol.unstack('ticker')

    if isinstance(ivol, pd.DataFrame):
        clean_ivol = pd.DataFrame(index=ivol.index, columns=ivol.columns)
    elif isinstance(ivol, pd.Series):
        clean_ivol = pd.DataFrame(index=ivol.index, columns=tickers)
    # normal_tests = pd.DataFrame(index=tickers, columns=['stat'])
    prob_dirty = pd.DataFrame(index=ivol.index, columns=clean_ivol.index)

    for ticker in tickers:

        print(ticker)

        try:

            z_t = pd.DataFrame(index=ivol.index)
            # z_t['r'] = stock_returns[ticker]
            if ref_ivol_ticker != ticker:
                z_t['rv'] = ivol[ref_ivol_ticker]
            # else:
                # z_t['r'] = stock_returns[ticker]

            df = BayesianTimeSeriesDataCleaner.clean_data(x=ivol[ticker],
                                                          z=z_t,
                                                          diff_type='level')

            clean_ivol[ticker] = df['x_clean']
            prob_dirty[ticker] = df['state_probs']
            # normal_tests.loc[ticker, 'stat'] = normaltest(r1.resid).statistic

        except:
            print('failed!')

    if len(orig_ivol.index.names) > 1:
        clean_ivol = clean_ivol.stack('ticker')
    orig_ivol.loc[clean_ivol.index] = clean_ivol.values
    clean_ivol = orig_ivol

    return clean_ivol, prob_dirty # , normal_tests


def _clean_implied_vol_data_one(stock_prices=None,
                                ivol=None,
                                ref_ivol=None,
                                res_com=5,
                                deg_f=2,
                                buffer_days=3,
                                pct_threshold=0.01,
                                calendar_name='UnitedStates'):

    ivol = ivol.copy(deep=True)
    ivol = ivol[np.isfinite(ivol)]

    df = pd.DataFrame(index=ivol.index,
                      columns=['px', 'ivol', 'ret', 'ivol_chg'])
    df['px'] = stock_prices
    df['ivol'] = ivol
    df = df[np.isfinite(df['px'])]
    df['ret'] = df['px'] / df['px'].shift(1) - 1
    df['ivol_chg'] = np.log(df['ivol'] / df['ivol'].shift(1))

    # Idea is that first we should predict ivol change based on stock chg
    # And then we should filter outliers from there

    x = df['ret']
    if ref_ivol is not None:
        ref_ivol = ref_ivol[np.isfinite(ref_ivol)]
        df['ref_ivol_chg'] = np.log(ref_ivol / ref_ivol.shift(1))
        x = df[['ret', 'ref_ivol_chg']]
    r1 = pd.ols(y=df['ivol_chg'], x=x)
    df['ivol_chg_res'] = r1.resid

    # Now... an outlier is a large unexpected change in ivol
    # And it's "post facto" an outlier if it then reverts back

    # So "probability of being an outlier" is some function of a large residual
    # followed by negative residuals

    # df['ivol_chg_res_fwd_ewm'] = df['ivol_chg_res'] \
    #                                  .iloc[::-1] \
    #                                  .ewm(com=res_com) \
    #                                  .mean() \
    #                                  .iloc[::-1] \
    #                                  .shift(-1) \
    #                                  * res_com

    df['ivol_chg_res_fwd_ewm'] = df['ivol_chg_res'] \
                                     .iloc[::-1] \
                                     .rolling(window=res_com) \
                                     .sum() \
                                     .iloc[::-1] \
                                     .shift(-1)

    max_date = np.max(df.index)
    df['ivol_chg_res_fwd_ewm'].loc[max_date] = 0

    tmp = df[['ivol_chg_res', 'ivol_chg_res_fwd_ewm']]
    tmpz = (tmp - tmp.mean()) / tmp.std()

    # Identify potential outliers
    from scipy.stats import t
    tmp_pr = t.pdf(tmpz['ivol_chg_res'], deg_f) \
             * t.pdf(tmpz['ivol_chg_res_fwd_ewm'], deg_f)
    tmp_pr = pd.DataFrame(data=tmp_pr, index=tmpz.index, columns=['tmp_pr'])
    tmp_pr_f = tmp_pr[tmp_pr['tmp_pr'] < pct_threshold]

    if len(tmp_pr_f) == 0:
        return ivol, tmp_pr, r1

    # Separate these into blocks
    tmp_pr_f['block'] = 0
    tmp_pr_f.loc[tmp_pr_f.index[0], 'block'] = 1
    dates = tmp_pr_f.index.get_level_values('date')
    for t in range(1, len(tmp_pr_f)):
        if dates[t] <= utils.workday(date=dates[t - 1],
                                     num_days=buffer_days,
                                     calendar_name=calendar_name):
            tmp_pr_f.loc[dates[t], 'block'] = tmp_pr_f.loc[dates[t-1], 'block']
        else:
            tmp_pr_f.loc[dates[t], 'block'] = tmp_pr_f.loc[
                                                dates[t-1], 'block'] + 1

    # NAN out that stuff
    ivol.loc[tmp_pr_f.index] = np.nan
    return ivol, tmp_pr, r1


def garman_klass_volatility(prices=None):

    """
    :param prices: DataFrame with columns ['high', 'low', 'open', 'close']
    :return: DataFrame
    """

    a = 0.5 * np.log(prices['high'] / prices['low']) ** 2
    b = (2 * np.log(2) - 1) * (np.log(prices['close'] / prices['open'])) ** 2
    x = float(constants.trading_days_per_year) / len(prices) * (a - b)
    vol = np.sqrt(x.sum())
    return vol


def _ensure_float(arr=None):
    output = [float(x) for x in arr]
    return output


def plot_normal(mu=None, sigma=None, x=None):
    mu = float(mu)
    sigma = float(sigma)
    if x == None:
        points = 5
        x = np.array(range(-3 * points, 3 * points)) * sigma / points + mu
    y = norm.pdf(x, mu, sigma)
    plt.plot(y, x)
    return pd.DataFrame(data=y, index=x, columns=['density'])


def compose_normals(mu1=None, mu2=None, sigma1=None, sigma2=None, x=None):
    [mu1, mu2, sigma1, sigma2] = _ensure_float([mu1, mu2, sigma1, sigma2])
    x1 = norm.pdf(x, mu1, sigma1)
    x2 = norm.pdf(x, mu2, sigma2)
    y = x1 * x2
    output = pd.DataFrame(index=x, data=y)
    output['x1'] = x1
    output['x2'] = x2
    output = output / output.sum() / (np.max(x) - np.min(x))
    return output


def plot_invgamma(a, b, x=None):
    a = float(a)
    b = float(b)
    if x == None:
        mean = b / (a - 1)
        stdev = (b ** 2 / ((a - 1) ** 2 * (a - 2))) ** 0.5
        points = 5
        x = np.array(range(-3 * points, 3 * points)) * stdev / points + mean
    y = invgamma.pdf(x, a, scale=b)
    plt.plot(y, x)
    return pd.DataFrame(data=y, index=x, columns=['density']), mean, stdev


def calc_forward_from_options(strikes, expiry_dates, option_types, option_prices):

    """
    :strikes:
    :expiry_dates:
    :option_types:
    :option_prices:
    :return:
    """

    # put-call parity: call + pv(cash) = put + stock + divs


def linear_setup(df, ind_cols, dep_col):
    '''
        Inputs: pandas Data Frame, list of strings for the independent variables,
        single string for the dependent variable
        Output: PyMC Model
    '''

    # model our intercept and error term as above
    b0 = pymc.Normal("b0", 0, 0.0001)
    err = pymc.Uniform("err", 0, 500)

    # initialize a NumPy array to hold our betas
    # and our observed x values
    b = np.empty(len(ind_cols), dtype=object)
    x = np.empty(len(ind_cols), dtype=object)

    # loop through b, and make our ith beta
    # a normal random variable, as in the single variable case
    for i in range(len(b)):
        b[i] = pymc.Normal("b" + str(i + 1), 0, 0.0001)

    # loop through x, and inform our model about the observed
    # x values that correspond to the ith position
    for i, col in enumerate(ind_cols):
        x[i] = pymc.Normal("x" + str(i + 1), 0, 1, value=np.array(df[col]),
                           observed=True)

    # as above, but use .dot() for 2D array (i.e., matrix) multiplication
    @pymc.deterministic
    def y_pred(b0=b0, b=b, x=x):
        return b0 + b.dot(x)

    # finally, "model" our observed y values as above
    y = pymc.Normal("y", y_pred, err, value=np.array(df[dep_col]),
                    observed=True)

    return pymc.Model(
        [b0, pymc.Container(b), err, pymc.Container(x), y, y_pred])

'''
--------------------------------------------------------------------------------
Volatility swaps
--------------------------------------------------------------------------------
'''


def _handle_dividend(div=0.0,
                     div_type='yield',
                     risk_free=0.0,
                     spot=None,
                     tenor_in_days=None):
    eps = 1e-7
    t = utils.numeric_cap_floor(
        x=tenor_in_days / constants.trading_days_per_year,
        floor=eps)

    if div_type == 'yield':
        pv_div = (1.0 - np.exp(-risk_free * t)) / (eps + risk_free) * div * t
    elif div_type == 'amount':
        pv_div = np.exp(-t * risk_free) * div
    else:
        pv_div = 0.0
    adj_spot = spot - pv_div

    return pv_div, adj_spot


def option_at_expiration(spot=None,
                         strike=None,
                         tenor_in_days=None,
                         risk_free=None,
                         ivol=None,
                         div=0.0,
                         div_type='yield',
                         d1=None,
                         option_type='c'):

    if isinstance(option_type, pd.Series):

        price = pd.Series(index=option_type.index)
        option_type = option_type.str.upper()

        call_ind = option_type.index[option_type.isin(['C', 'CALL'])]
        price.loc[call_ind] = np.max(0, spot.loc[call_ind] - strike.loc[call_ind])

        put_ind = option_type.index[option_type.isin(['P', 'PUT'])]
        price.loc[put_ind] = np.max(0, strike.loc[put_ind] - spot.loc[put_ind])

    else:
        if option_type.upper() in (['C', 'CALL']):
            price = np.max(0, spot - strike)
        elif option_type.upper() in (['P', 'PUT']):
            price = np.max(0, strike - spot)

    return price


def black_scholes_price(spot=None,
                        strike=None,
                        tenor_in_days=None,
                        risk_free=None,
                        ivol=None,
                        div=0.0,
                        div_type='yield',
                        d1=None,
                        option_type='c'):

    eps = 1e-7
    t = utils.numeric_cap_floor(
        x=tenor_in_days / constants.trading_days_per_year,
        floor=eps)
    pv_div, adj_spot = _handle_dividend(div=div,
                                        div_type=div_type,
                                        risk_free=risk_free,
                                        spot=spot,
                                        tenor_in_days=tenor_in_days)

    if d1 is None:
        d1, d2 = black_scholes_d1(spot=spot,
                                  strike=strike,
                                  tenor_in_days=tenor_in_days,
                                  risk_free=risk_free,
                                  ivol=ivol,
                                  div=div,
                                  div_type=div_type)

    if isinstance(option_type, pd.Series):

        price = pd.Series(index=option_type.index)
        option_type = option_type.str.upper()

        d1 = pd.to_numeric(d1)
        d2 = pd.to_numeric(d2)

        call_ind = option_type.index[option_type.isin(['C', 'CALL'])]
        price.loc[call_ind] = adj_spot.loc[call_ind] \
            * norm.cdf(d1.loc[call_ind]) \
            - np.exp(-risk_free.loc[call_ind] * t.loc[call_ind])\
              * strike.loc[call_ind] * norm.cdf(d2.loc[call_ind])

        put_ind = option_type.index[option_type.isin(['P', 'PUT'])]
        price.loc[put_ind] = strike.loc[put_ind] \
            * np.exp(-risk_free.loc[put_ind]
            * t.loc[put_ind]) * norm.cdf(-d2.loc[put_ind]) \
            - spot.loc[put_ind] * norm.cdf(-d1.loc[put_ind])

    else:
        if option_type.upper() in (['C', 'CALL']):
            price = adj_spot * norm.cdf(d1) \
                    - np.exp(-risk_free * t) * strike * norm.cdf(d2)
        elif option_type.upper() in (['P', 'PUT']):
            price = strike * np.exp(-risk_free * t) * norm.cdf(-d2) \
                    - spot * norm.cdf(-d1)

    return price


def put_call_parity(input_price=None,
                    option_type='c',
                    spot=None,
                    strike=None,
                    div=0.0,
                    div_type='yield',
                    risk_free=None,
                    tenor_in_days=None):

    eps = 1e-7
    t = utils.numeric_cap_floor(
        x=tenor_in_days / constants.trading_days_per_year,
        floor=eps)
    pv_div, adj_spot = _handle_dividend(div=div,
                                        div_type=div_type,
                                        risk_free=risk_free,
                                        spot=spot,
                                        tenor_in_days=tenor_in_days)

    # cash plus call plus divs equals put plus stock
    if option_type.upper() in (['C', 'CALL']):
        other_price = input_price + spot \
                      - np.exp(risk_free * t) * strike - pv_div
    elif option_type.upper() in (['P', 'PUT']):
        other_price = np.exp(risk_free * t) * strike + pv_div - spot
    return other_price


def black_scholes_delta(spot=None,
                        strike=None,
                        tenor_in_days=None,
                        risk_free=None,
                        ivol=None,
                        div=0.0,
                        div_type='yield',
                        d1=None,
                        option_type=None):

    eps = 1e-7
    t = utils.numeric_cap_floor(
        x=tenor_in_days / constants.trading_days_per_year,
        floor=eps)
    pv_div, adj_spot = _handle_dividend(div=div,
                                        div_type=div_type,
                                        risk_free=risk_free,
                                        spot=spot,
                                        tenor_in_days=tenor_in_days)
    div_yield = pv_div / spot / t

    if d1 is None:
        d1, d2 = black_scholes_d1(spot=spot,
                                  strike=strike,
                                  tenor_in_days=tenor_in_days,
                                  risk_free=risk_free,
                                  ivol=ivol,
                                  div=div,
                                  div_type=div_type)

    if isinstance(option_type, pd.Series):

        delta = pd.Series(index=option_type.index)
        option_type = option_type.str.upper()

        call_ind = option_type.index[option_type.isin(['C', 'CALL'])]
        delta.loc[call_ind] = np.exp(-div_yield.loc[call_ind]
            * t.loc[call_ind]) * norm.cdf(d1.loc[call_ind])

        put_ind = option_type.index[option_type.isin(['P', 'PUT'])]
        delta.loc[put_ind] = -np.exp(-div_yield.loc[put_ind]
            * t.loc[put_ind]) * norm.cdf(-d1.loc[put_ind])

    else:
        if option_type.upper() in (['C', 'CALL']):
            delta = np.exp(-div_yield * t) * norm.cdf(d1)
        elif option_type.upper() in (['P', 'PUT']):
            delta = -np.exp(-div_yield * t) * norm.cdf(-d1)

    return delta


def black_scholes_vega(spot=None,
                       strike=None,
                       tenor_in_days=None,
                       risk_free=None,
                       ivol=None,
                       div=0.0,
                       div_type='yield',
                       d1=None,
                       option_type=None):

    eps = 1e-7
    t = utils.numeric_cap_floor(
        x=tenor_in_days / constants.trading_days_per_year,
        floor=eps)
    if div_type == 'yield':
        div_yield = div
    elif div_type == 'amount':
        div_yield = div / spot / t

    if d1 is None:
        d1, d2 = black_scholes_d1(spot=spot,
                                  strike=strike,
                                  tenor_in_days=tenor_in_days,
                                  risk_free=risk_free,
                                  ivol=ivol,
                                  div=div,
                                  div_type=div_type)

    vega = spot * np.exp(-div_yield * t) * norm.pdf(d1) * np.sqrt(t) / 100.0
    return vega


def black_scholes_d1(spot=None,
                     strike=None,
                     tenor_in_days=None,
                     risk_free=None,
                     ivol=None,
                     div=None,
                     div_type='yield'):

    eps = 1e-7
    t = utils.numeric_cap_floor(
        x=tenor_in_days / constants.trading_days_per_year,
        floor=eps)
    pv_div, adj_spot = _handle_dividend(div=div,
                                        div_type=div_type,
                                        risk_free=risk_free,
                                        spot=spot,
                                        tenor_in_days=tenor_in_days)

    d1 = (np.log(adj_spot / strike) + (risk_free + ivol ** 2.0 / 2.0) * t) \
         / (ivol * t ** 0.5)

    d2 = d1 - ivol * t ** 0.5

    d1 = pd.to_numeric(d1)
    d2 = pd.to_numeric(d2)

    return d1, d2


def black_scholes_moneyness_from_delta(call_delta=None,
                                       tenor_in_days=None,
                                       ivol=None,
                                       risk_free=None,
                                       div_yield=None):

    """
    :param call_delta: this can be a scalar or an array-like. If the latter,
    ivol needs to have call delta as its columns
    :param tenor_in_days: this can be an integer or a Series with index that
    matches ivol's index
    :param ivol: this can be a scalar, a series, or a DataFrame indexed on
    date and/or ticker, with columns = call delta
    :param risk_free: can be scalar or series
    :param div_yield: can be scalar or series
    :return: scalar, series or DataFrame depending on inputs
    """

    if isinstance(call_delta, list):
        call_delta = np.array(call_delta)

    put_delta = 1.0 - call_delta
    eps = 1e-7
    t = utils.numeric_cap_floor(x=tenor_in_days/constants.trading_days_per_year,
                                floor=eps)

    if isinstance(tenor_in_days, pd.Series) and isinstance(ivol, pd.DataFrame):
        m = pd.DataFrame(index=ivol.index, columns=ivol.columns)
        for col in ivol.columns:
            m[col] = np.exp(ivol[col] * t ** 0.5 * norm.ppf(
                (1 - col) * np.exp(div_yield * t))
                     - (risk_free - div_yield - ivol[col] ** 2.0 / 2.0) * t)
    else:
        m = np.exp(ivol * t ** 0.5 * norm.ppf(put_delta * np.exp(div_yield * t))
                     - (risk_free - div_yield - ivol ** 2.0 / 2.0) * t)
    return m


def round_to_sig_dig(x=None, num_digits=1):
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        d = np.floor(1 - num_digits + np.log10(np.abs(x)))
        output = x.round(-d.astype(int))
    else:
        output = round(x, -int(np.floor(1 - num_digits + np.log10(np.abs(x)))))
    return output

'''
--------------------------------------------------------------------------------
Volatility swaps
--------------------------------------------------------------------------------
'''


def volswap_market_value(iv=None,
                         rv=None,
                         strike=None,
                         days_elapsed=None,
                         total_days=None,
                         date=None,
                         start_date=None,
                         maturity_date=None,
                         calendar_name='UnitedStates'):

    # If maturity date and trade date are provided, those take precedence
    if start_date is not None and maturity_date is not None and date is not None:
        days_to_maturity, days_elapsed, total_days \
            = utils.get_days_from_maturity(
                start_date=start_date,
                maturity_date=maturity_date,
                date=date,
                calendar_name=calendar_name)

    # Weighted average of implied and realized
    if isinstance(days_elapsed, int):
        days_elapsed = float(days_elapsed)
    elif utils.is_iterable(days_elapsed):
        days_elapsed = days_elapsed.astype(float)
    rlzd_weight = days_elapsed / total_days

    ev = (iv ** 2 * (1 - rlzd_weight) + rv ** 2 * rlzd_weight) ** 0.5
    return ev - strike


def volswap_daily_pnl(daily_return=None,
                      iv=None,
                      rv=None,
                      strike=None,
                      days_elapsed=None,
                      total_days=None,
                      date=None,
                      start_date=None,
                      maturity_date=None,
                      calendar_name='UnitedStates'):

    """
    This calculates the (gamma-theta-only) daily PNL of a volswap
    It assumes no change in IV (obviously can extend)
    :param daily_return:
    :param iv:
    :param rv:
    :param strike:
    :param days_elapsed:
    :param total_days:
    :param date:
    :param start_date:
    :param maturity_date:
    :param calendar_name:
    :return: double or DataFrame
    """

    # If maturity date and trade date are provided, those take precedence
    if start_date is not None and maturity_date is not None and date is not None:
        days_to_maturity, days_elapsed, total_days \
            = utils.get_days_from_maturity(
                start_date=start_date,
                maturity_date=maturity_date,
                date=date,
                calendar_name=calendar_name)
        if start_date > date:
            if utils.is_iterable(iv):
                theta = pd.DataFrame(index=iv.index, columns=['theta'])
                theta['theta'] = 0
            else:
                theta = 0
            return theta

    mv_0 = volswap_market_value(iv=iv,
                                rv=rv,
                                strike=strike,
                                days_elapsed=days_elapsed,
                                total_days=total_days,
                                calendar_name=calendar_name)

    rv_1 = (days_elapsed / (days_elapsed + 1) * rv ** 2
         + constants.trading_days_per_year * daily_return ** 2) ** 0.5

    mv_1 = volswap_market_value(iv=iv,
                                rv=rv_1,
                                strike=strike,
                                days_elapsed=days_elapsed + 1,
                                total_days=total_days,
                                calendar_name=calendar_name)

    pnl = mv_1 - mv_0
    return pnl


def volswap_gamma(iv=None,
                  rv=None,
                  strike=None,
                  days_elapsed=None,
                  total_days=None,
                  date=None,
                  start_date=None,
                  maturity_date=None,
                  calendar_name='UnitedStates'):

    x=1


def volswap_theta(iv=None,
                  rv=None,
                  strike=None,
                  days_elapsed=None,
                  total_days=None,
                  date=None,
                  start_date=None,
                  maturity_date=None,
                  calendar_name='UnitedStates'):

    if utils.is_iterable(iv):
        daily_return = pd.DataFrame(index=rv.index, columns=['returns'])
        daily_return['returns'] = 0
        daily_return = daily_return['returns']
    else:
        daily_return = 0

    theta = volswap_daily_pnl(daily_return=daily_return,
                              iv=iv,
                              rv=rv,
                              strike=strike,
                              days_elapsed=days_elapsed,
                              total_days=total_days,
                              date=date,
                              start_date=start_date,
                              maturity_date=maturity_date,
                              calendar_name=calendar_name)
    return theta
