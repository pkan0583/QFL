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
                  & (np.isfinite(generic_futures_prices.loc[date]))

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

    if volatilities:
        constant_maturity_futures_prices **= 0.5

    return constant_maturity_futures_prices


def clean_implied_vol_data(tickers=None,
                           stock_prices=None,
                           ivol=None,
                           ref_ivol_ticker=None,
                           res_com=5,
                           deg_f=4,
                           buffer_days=3,
                           pct_threshold=0.0001,
                           calendar_name='UnitedStates'):

    orig_ivol = ivol.copy(deep=True)
    ivol = ivol.copy(deep=True)
    stock_prices = stock_prices.copy(deep=True)

    if isinstance(ivol, pd.DataFrame):
        clean_ivol = pd.DataFrame(index=ivol.index, columns=ivol.columns)
    elif isinstance(ivol, pd.Series):
        clean_ivol = pd.DataFrame(index=ivol.index, columns=tickers)
    normal_tests = pd.DataFrame(index=tickers, columns=['stat'])

    for ticker in tickers:

        print(ticker)

        try:

            riv = ivol[ref_ivol_ticker]
            if ref_ivol_ticker == ticker:
                riv = None

            clean_ivol_tmp, tmp_pr, r1 = _clean_implied_vol_data_one(
                stock_prices=stock_prices[ticker],
                ivol=ivol[ticker],
                ref_ivol=riv,
                pct_threshold=pct_threshold,
                buffer_days=buffer_days,
                res_com=res_com,
                deg_f=deg_f,
                calendar_name=calendar_name)

            clean_ivol_tmp = clean_ivol_tmp[np.isfinite(clean_ivol_tmp)]

            clean_ivol[ticker], tmp_pr_, r1_ = _clean_implied_vol_data_one(
                stock_prices=stock_prices[ticker],
                ivol=clean_ivol_tmp,
                ref_ivol=riv,
                pct_threshold=pct_threshold,
                buffer_days=buffer_days,
                res_com=res_com,
                deg_f=deg_f,
                calendar_name=calendar_name)

            normal_tests.loc[ticker, 'stat'] = normaltest(r1.resid).statistic

        except:
            print('failed!')

    orig_ivol.loc[clean_ivol.index] = clean_ivol.values
    clean_ivol = orig_ivol

    return clean_ivol, normal_tests


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
