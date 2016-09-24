import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdata

from sklearn.decomposition import PCA, KernelPCA,  FactorAnalysis
from sklearn import cluster, covariance, manifold
from scipy import interpolate

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import struct

import qfl.utilities.basic_utilities as utils
import qfl.utilities.statistics as stats


class FundamentalMacroModel(object):

    default_start_date = dt.datetime(1980, 1, 1)
    settings = None
    raw_data = None
    data = None
    calc = None

    @classmethod
    def run(cls, start_date=None):

        ffill_limit = 3
        diff_window = 1
        diff_threshold = 0.875

        if cls.settings is None:
            macro_data, settings, raw_macro_data = cls.load_fundamental_data()
        else:
            settings = cls.settings
            macro_data = cls.data.macro_data
            raw_macro_data = cls.raw_data

        macro_data, macro_data_1d, macro_data_3d, macro_data_6d, macro_data_ma = \
            cls.process_factor_model_data(raw_macro_data=raw_macro_data,
                                          macro_data=macro_data,
                                          settings=settings)

        # AR1
        ar1_coefs = cls.compute_ar_coefs(macro_data=macro_data,
                                         macro_data_1d=macro_data_1d,
                                         macro_data_3d=macro_data_3d,
                                         macro_data_6d=macro_data_6d,
                                         macro_data_ma=macro_data_ma,
                                         settings=settings)

        macro_data_stationary = cls.process_data_by_ar(
            macro_data=macro_data,
            ar1_coefs=ar1_coefs,
            ar1_threshold=diff_threshold,
            diff_window=diff_window)

        # Z-scores
        macro_data_z = ((macro_data_stationary - macro_data_stationary.mean())
                        / macro_data_stationary.std()).fillna(method='ffill',
                                                              limit=ffill_limit)
        components, pca_factors, pca_obj = cls.compute_factors(
            macro_data_z=macro_data_z,
            settings=settings)
        components, pca_factors = cls.process_factors(
            components=components,
            pca_factors=pca_factors,
            settings=settings)

        # MA Z-scores
        macro_data_maz = (
        (macro_data_ma - macro_data_ma.mean()) / macro_data_ma.std()
            ).fillna(method='ffill', limit=ffill_limit)
        components_ma, pca_factors_ma, pca_obj_ma = cls.compute_factors(
            macro_data_z=macro_data_maz, settings=settings)
        components_ma, pca_factors_ma = cls.process_factors(
            components=components_ma, pca_factors=pca_factors_ma,
            settings=settings)

        cls.calc = struct
        cls.calc.macro_data_stationary = macro_data_stationary
        cls.calc.ar1_coefs = ar1_coefs
        cls.calc.components = components
        cls.calc.pca_factors = pca_factors
        cls.calc.components_ma = components_ma
        cls.calc.pca_factors_ma = pca_factors_ma

        return cls.calc

    @classmethod
    def load_fundamental_data(cls, start_date=None):

        if start_date is None:
            start_date = cls.default_start_date

        # Settings
        settings = struct
        settings.start_date = start_date
        settings.config_filename = 'qfl/macro/macro_model_cfg.csv'
        settings.gdp_fieldname = 'GDP'
        settings.cpi_fieldname = 'CPILFESL'
        settings.pop_fieldname = 'POP'

        # Read configuration
        settings.config_table = pd.read_csv(settings.config_filename)

        # Active filter
        settings.config_table = settings.config_table[
            settings.config_table['ACTIVE'] == 1]

        # Fields and field names
        settings.data_fields = settings.config_table['FRED_CODE'].tolist()
        settings.data_fieldnames = settings.config_table['SERIES_NAME'].tolist()
        settings.use_log = settings.config_table['USE_LOG'].tolist()
        settings.ma_length = settings.config_table['MA_LENGTH'].tolist()
        settings.div_gdp = settings.config_table['DIV_GDP'].tolist()
        settings.div_pop = settings.config_table['DIV_POP'].tolist()
        settings.div_cpi = settings.config_table['DIV_CPI'].tolist()

        # Load data
        raw_macro_data = pdata.get_data_fred(settings.data_fields, start_date)
        macro_data = raw_macro_data.copy()

        # Get GDP data and fill in divisors
        macro_data = cls._get_gdp_and_process(macro_data=macro_data,
                                              settings=settings)

        # Process retail data (joining two series)
        macro_data = cls._process_retail_data(macro_data=macro_data,
                                              settings=settings)

        # Get claims data (weekly)
        macro_data, settings = cls._add_claims_data(macro_data=macro_data,
                                                    settings=settings)

        # Clean data
        macro_data = cls._clean_data(macro_data=macro_data,
                                     settings=settings)

        # Get series start and end dates
        settings.series_start_dates = pd.DataFrame(index=settings.data_fields,
                                                   columns=['date'])
        settings.series_end_dates = pd.DataFrame(index=settings.data_fields,
                                                 columns=['date'])
        for data_field in settings.data_fields:
            finite_data = macro_data[data_field][np.isfinite(macro_data[data_field])]
            settings.series_start_dates.loc[data_field] = finite_data.index.min()
            settings.series_end_dates.loc[data_field] = finite_data.index.max()
        settings.common_start_date = settings.series_start_dates.values.max()
        settings.common_end_date = settings.series_end_dates.values.min()

        cls.settings = settings
        cls.data = struct
        cls.data.macro_data = macro_data
        cls.raw_data = raw_macro_data

        return macro_data, settings, raw_macro_data

    @classmethod
    def _get_gdp_and_process(cls, macro_data=None, settings=None):

        gdp_data = pdata.get_data_fred(settings.gdp_fieldname, settings.start_date)
        macro_data[settings.gdp_fieldname] = gdp_data

        pop_data = pdata.get_data_fred(settings.pop_fieldname, settings.start_date)
        macro_data[settings.pop_fieldname] = pop_data

        cols = [settings.gdp_fieldname, settings.pop_fieldname]

        macro_data[cols] = macro_data[cols].fillna(method='ffill')
        macro_data[cols] = macro_data[cols].fillna(method='bfill')

        return macro_data

    @classmethod
    def _process_retail_data(cls, macro_data=None, settings=None):

        # Filling in retail data: join two series
        if 'RSXFS' in macro_data.columns:
            retail_series_join_date = dt.datetime(1992, 1, 1)
            other_retail_sales = pdata.get_data_fred('RETAIL', settings.start_date)
            levels_ratio = other_retail_sales['RETAIL'] \
                [other_retail_sales.index == retail_series_join_date] \
                / macro_data['RSXFS'][macro_data.index == retail_series_join_date][0]
            other_retail_sales = other_retail_sales['RETAIL'] / levels_ratio[0]
            macro_data['d'] = other_retail_sales
            macro_data['RSXFS'][np.isnan(macro_data['RSXFS'])] \
                = macro_data['d'][np.isnan(macro_data['RSXFS'])]
            del macro_data['d']
        return macro_data

    @classmethod
    def _add_claims_data(cls, macro_data=None, settings=None):

        # Handle claims data
        claims = pdata.get_data_fred('IC4WSA', settings.start_date)
        claims = claims.resample('M').last()
        claims.index = claims.index + pd.tseries.offsets.BDay(1)
        tmp = [dt.datetime(claims.index[t].year, claims.index[t].month, 1)
               for t in range(0, len(claims))]
        claims.index = pd.DatetimeIndex(tmp)
        macro_data['Claims'] = claims
        settings.data_fields = settings.data_fields + ['Claims']
        settings.data_fieldnames = settings.data_fieldnames + ['Jobless Claims']
        settings.use_log += [0]
        settings.div_gdp += [0]
        settings.div_pop += [1]
        settings.div_cpi += [0]
        settings.ma_length += [12]

        return macro_data, settings

    @classmethod
    def _clean_data(cls, macro_data=None, settings=None):

        # manual override
        if 'UMCSENT' in settings.data_fields:
            macro_data['UMCSENT'][macro_data.index == dt.datetime(2015, 8, 1)] = 91.9
            macro_data['UMCSENT'][macro_data.index == dt.datetime(2015, 9, 1)] = 87.2
            macro_data['UMCSENT'][macro_data.index == dt.datetime(2015, 10, 1)] = 90.0
            macro_data['UMCSENT'][macro_data.index == dt.datetime(2015, 11, 1)] = 91.3
            macro_data['UMCSENT'][macro_data.index == dt.datetime(2015, 12, 1)] = 92.6

        if 'DTBTM' in settings.data_fields:
            tmp = macro_data['DTBTM'][macro_data.index == dt.datetime(2015, 10, 1)]
            macro_data['DTBTM'][macro_data.index == dt.datetime(2015, 11, 1)] = tmp[0]
            macro_data['DTBTM'][macro_data.index == dt.datetime(2015, 12, 1)] \
                = macro_data['DTBTM'][macro_data.index == dt.datetime(2015, 11, 1)][0]

        return macro_data

    @classmethod
    def process_factor_model_data(cls,
                                  raw_macro_data=None,
                                  macro_data=None,
                                  settings=None):

        # Make adjustments
        for i in range(0, len(settings.data_fields)):
            if settings.use_log[i] == 1:
                macro_data[settings.data_fields[i]] = \
                    np.log(macro_data[settings.data_fields[i]])
            if settings.div_gdp[i] == 1:
                macro_data[settings.data_fields[i]] \
                    /= macro_data[settings.gdp_fieldname]
            if settings.div_pop[i] == 1:
                macro_data[settings.data_fields[i]] /= macro_data[settings.pop_fieldname]
            if settings.div_cpi[i] == 1:
                macro_data[settings.data_fields[i]] /= raw_macro_data[settings.cpi_fieldname]

        # First difference
        macro_data_1d = macro_data.diff(1)

        # Manual data override
        if 'DTBTM' in settings.data_fields:
            macro_data_1d['DTBTM'][macro_data_1d.index
                                   == dt.datetime(2010, 12, 1)] \
                = -0.05

        # Three and six month differences
        macro_data_3d = macro_data_1d.rolling(window=3, center=False).mean()
        macro_data_6d = macro_data_1d.rolling(window=6, center=False).mean()

        # Moving averages (different lookback for each series
        # To roughly harmonize autoregressive tendencies
        macro_data_ma = pd.DataFrame()
        for i in range(0, len(settings.data_fieldnames)):
            macro_data_ma[settings.data_fields[i]] = \
                macro_data_1d[settings.data_fields[i]] \
                .rolling(window=settings.ma_length[i],
                         center=False) \
                .mean()

        return macro_data, macro_data_1d, \
               macro_data_3d, macro_data_6d, macro_data_ma

    @classmethod
    def compute_ar_coefs(cls,
                         macro_data=None,
                         macro_data_1d=None,
                         macro_data_3d=None,
                         macro_data_6d=None,
                         macro_data_ma=None,
                         settings=None):

        reg_outputs = dict()
        ar1_coefs = np.zeros((len(settings.data_fields), 5))
        predictedData = pd.DataFrame()
        cleanData = macro_data_ma.copy()
        cleanData = cleanData[cleanData.index > settings.common_start_date]
        days_since_start = (macro_data.index - macro_data.index[0]).days
        days_since_start = pd.DataFrame(days_since_start)
        days_since_start.index = macro_data.index
        for i in range(0, len(settings.data_fields)):

            # Trend coefficients
            tmp = pd.ols(y=macro_data[settings.data_fields[i]],
                         x=days_since_start[0])

            tmp = pd.ols(y=macro_data[settings.data_fields[i]],
                         x=macro_data[settings.data_fields[i]].shift(1))
            ar1_coefs[i, 0] = (tmp.beta['x'])

            tmp = pd.ols(y=macro_data_1d[settings.data_fields[i]],
                         x=macro_data_1d[settings.data_fields[i]].shift(1))
            ar1_coefs[i, 1] = (tmp.beta['x'])

            tmp = pd.ols(y=macro_data_3d[settings.data_fields[i]],
                         x=macro_data_3d[settings.data_fields[i]].shift(1))
            ar1_coefs[i, 2] = (tmp.beta['x'])

            tmp = pd.ols(y=macro_data_6d[settings.data_fields[i]],
                         x=macro_data_6d[settings.data_fields[i]].shift(1))
            ar1_coefs[i, 3] = (tmp.beta['x'])

            tmp = pd.ols(y=macro_data_ma[settings.data_fields[i]],
                         x=macro_data_ma[settings.data_fields[i]].shift(1))
            ar1_coefs[i, 4] = (tmp.beta['x'])

        ar1_coefs = pd.DataFrame(data=ar1_coefs,
                                 index=settings.data_fields,
                                 columns=['levels', 'd1', 'd3', 'd6', 'dma'])
        ar1_coefs['ma_length'] = settings.ma_length

        return ar1_coefs

    @classmethod
    def process_data_by_ar(cls,
                           macro_data=None,
                           ar1_coefs=None,
                           ar1_threshold=0.875,
                           diff_window=1):

        # Any series with an AR above the threshold is differenced
        macro_data_stationary = macro_data.copy(deep=True)
        ind = ar1_coefs.index[ar1_coefs['levels'] > ar1_threshold]
        for i in ind:
            macro_data_stationary[i] = macro_data[i].diff(diff_window)
        return macro_data_stationary

    @classmethod
    def compute_factors(cls, macro_data_z=None, settings=None, num_factors=4):

        # Minimum dataset with all observations
        pca_data = macro_data_z[settings.data_fields].copy(deep=True)
        for field in settings.data_fields:
            pca_data = pca_data[np.isfinite(pca_data[field])]

        # Estimate PCA
        princomp = PCA(n_components=num_factors)
        s1 = princomp.fit(pca_data.values).transform(pca_data)
        components = pd.DataFrame(data=princomp.components_.transpose(),
                                  index=settings.data_fieldnames)
        pca_factors = pd.DataFrame(data=s1, index=pca_data.index)

        return components, pca_factors, princomp

    @classmethod
    def process_factors(cls, components=None, pca_factors=None, settings=None):

        # F1 is growth
        if 'Unemployment Rate' in settings.data_fieldnames:
            if components.loc['Unemployment Rate', 0] > 0:
                components[0] = -components[0]
                pca_factors[0] = -pca_factors[0]

        # F2 is inflation/credit
        if 'Total Consumer Credit' in settings.data_fieldnames:
            if components.loc['Total Consumer Credit', 1] < 0:
                components[1] = -components[1]
                pca_factors[1] = -pca_factors[1]

        # F3 is real estate
        if 'Total Construction Spending' in settings.data_fieldnames:
            if components.loc['Total Construction Spending', 2] < 0:
                components[2] = -components[2]
                pca_factors[2] = -pca_factors[2]

        # F4 is the labor market
        if 'Unemployment Rate' in settings.data_fieldnames:
            if components.loc['Unemployment Rate', 3] > 0:
                components[3] = -components[3]
                pca_factors[3] = -pca_factors[3]

        return components, pca_factors

    @classmethod
    def retrieve_recession_data(cls, macro_data=None, start_date=None):

        recession_data = pdata.get_data_fred('USRECD', start=start_date)
        recession_start_dates = []
        recession_end_dates = []
        for i in range(1, len(recession_data)):
            if (recession_data.values[i][0] == 0) & (
                        recession_data.values[i - 1][0] == 1):
                recession_end_dates.append(recession_data.index[i])
            elif (recession_data.values[i][0] == 1) & (
                        recession_data.values[i - 1][0] == 0):
                recession_start_dates.append(recession_data.index[i])

        # Assign zeros to recession dates since last data point
        tmp_date = recession_data.index[len(recession_data) - 1]
        df = pd.DataFrame(macro_data.index[macro_data.index > tmp_date])
        df.index = macro_data.index[macro_data.index > tmp_date]
        df['USRECD'] = 0
        recession_data = pd.DataFrame.append(recession_data, df)
        del recession_data['DATE']

        return recession_data, recession_start_dates, recession_end_dates

    @classmethod
    def estimate_recession_probability(cls,
                                       macro_data=None,
                                       macro_indicator=None,
                                       start_date=None):
        import scipy
        from statsmodels.discrete.discrete_model import Logit, Probit
        recession_data, recession_start_dates, recession_end_dates = \
            cls.retrieve_recession_data(macro_data=macro_data,
                                        start_date=start_date)

        # Settings for the analysis
        recession_prediction_window = 252
        windows = [1, 3, 6, 9, 12, 18, 24]
        halflifes = [0.5, 1, 2, 3, 6, 9, 12]

        r2 = []
        ll = []
        recession_probs = pd.DataFrame(index=macro_indicator.index,
                                       columns=halflifes)
        recession_in_window = pd.DataFrame(index=macro_indicator.index)

        tmp = recession_data.rolling(window=recession_prediction_window,
                                     center=False).sum()
        tmp2 = tmp.shift(-recession_prediction_window)
        recession_in_window['RIW'] = np.minimum(1, tmp2)
        logits = []
        regs = []

        for i in range(0, len(windows)):
            logit_iv = pd.DataFrame(index=recession_data.index)
            logit_iv['Recession'] = recession_data
            logit_iv['RIW'] = recession_in_window['RIW']

            # Exponential moving average of the macro indicator
            window_ma = macro_indicator.ewm(halflife=halflifes[i], ignore_na=False,
                                            min_periods=0, adjust=True).mean()

            logit_iv['F1M'] = window_ma.copy()
            logit_iv['LF1'] = window_ma.shift(windows[i])
            logit_iv['DF1'] = window_ma.diff(halflifes[i] * 2)
            logit_iv['INT'] = logit_iv['F1M'] * logit_iv['DF1']
            logit_iv = logit_iv[(~np.isnan(logit_iv['DF1']))
                                & (~np.isnan(logit_iv['F1M']))]
            logit_iv['RIW'][np.isnan(logit_iv['RIW'])] = 0

            # tmp = Logit(logit_iv['RIW'], logit_iv[['F1M', 'DF1', 'INT']])
            tmp = Probit(logit_iv['RIW'], logit_iv['F1M'])
            # tmp = linear_model.OLS(logit_iv['RIW'], pd.DataFrame(logit_iv['F1M']))
            result = tmp.fit()
            predictions = result.fittedvalues[~np.isnan(result.fittedvalues)]
            logits.append(result)

            ll.append(result.llr)
            # recession_probs[str(i)] = np.exp(predictions) / (1 + np.exp(predictions))
            recession_probs.loc[logit_iv.index, halflifes[i]] \
                = scipy.stats.norm.cdf(predictions)

    @classmethod
    def prepare_volatility_data(cls,
                                ticker='^GSPC',
                                start_date=None,
                                exclude87=False,
                                exclude_start_date=dt.datetime(1987, 10, 19),
                                exclude_end_date=dt.datetime(1987, 10, 21),
                                cap_factor=2.5,
                                vol_windows=[21, 63, 126, 252],
                                volatility_indicator='Volatility252'):

            # Get data
            prices = pdata.DataReader(ticker,
                                      data_source='yahoo',
                                      start=start_date,
                                      end=dt.datetime.today())
            prices = prices['Adj Close']
            returns = np.log(prices) - np.log(prices).shift(1)

            # Exclude 1987 to reduce noise
            if exclude87:
                returns[(returns.index >= exclude_start_date)
                      & (returns.index <= exclude_end_date)] = 0

            variable_names = []
            volatility = pd.DataFrame()
            volatility_lead = pd.DataFrame()
            for i in range(0, len(vol_windows)):
                variable_names.append('Volatility' + str(vol_windows[i]))
                volatility[variable_names[i]] = \
                    np.sqrt(252.0 * returns.rolling(
                        window=vol_windows[i],
                        center=False).var())
                volatility_lead[variable_names[i]] = volatility[
                    variable_names[i]].shift(-vol_windows[i])

            indVars = pd.DataFrame()
            indVars['Volatility21'] = volatility['Volatility21']
            indVars['Volatility63'] = volatility['Volatility63']
            indVars['Volatility252'] = volatility['Volatility252']

            # Apply caps to daily returns to reduce noise
            sp500_returns_capped = returns.copy()
            ind = (np.abs(returns) > cap_factor *
                   indVars['Volatility252'] / np.sqrt(252))
            sp500_returns_capped[ind] = np.sign(returns[ind]) \
                                        * cap_factor * indVars[
                                            'Volatility252'] / np.sqrt(252)

            variable_names = []
            capped_volatility = pd.DataFrame()
            capped_volatility_lead = pd.DataFrame()
            for i in range(0, len(vol_windows)):
                variable_names.append('Volatility' + str(vol_windows[i]))
                capped_volatility[variable_names[i]] = np.sqrt(
                    252.0 * sp500_returns_capped.rolling(
                        window=vol_windows[i],
                        center=False).var())
                capped_volatility_lead[variable_names[i]] = \
                    volatility[variable_names[i]].shift(-vol_windows[i])

            ind_vars_df = pd.DataFrame()
            ind_vars_df[0] = capped_volatility[volatility_indicator].copy()

            vol_ac_reg = pd.ols(y=capped_volatility_lead[volatility_indicator],
                                x=ind_vars_df)
            residual_volatility = vol_ac_reg.resid

            return capped_volatility, capped_volatility_lead, residual_volatility

    @classmethod
    def prepare_nonparametric_analysis(cls,
                                       capped_volatility_lead=None,
                                       capped_volatility=None,
                                       residual_volatility=None,
                                       macro_indicator=None,
                                       volatility_indicator='Volatility252',
                                       start_date=None,
                                       short_alpha=0.50,
                                       long_alpha=0.15):

            # time weights for recency weighting
            time_weight_param = 1000
            level_grid_min = -2.5
            level_grid_max = 2.5
            change_grid_min = -2.5
            change_grid_max = 2.5

            dr = pd.date_range(start_date, dt.datetime.today())
            npd = pd.DataFrame(index=dr)
            npd['Volatility'] = capped_volatility_lead[volatility_indicator].copy()
            npd['residual_volatility'] = residual_volatility
            npd['TrailingVolatility'] = capped_volatility[volatility_indicator].copy()

            # Make sure everything lines up on business days
            npd = npd.fillna(method='ffill')

            npd['F1'] = macro_indicator

            # Reduce to monthly dataset
            npd = npd[~np.isnan(npd['F1'])]

            # Lags and changes
            macro_indicator_ma = macro_indicator.ewm(alpha=short_alpha,
                                                     ignore_na=True,
                                                     min_periods=0,
                                                     adjust=True).mean()

            macro_indicator_ma_long = macro_indicator.ewm(alpha=long_alpha,
                                                          ignore_na=True,
                                                          min_periods=0,
                                                          adjust=True).mean()

            npd['DF1'] = macro_indicator_ma - macro_indicator_ma_long
            npd['LF1'] = macro_indicator_ma_long
            npd['F1M'] = macro_indicator_ma
            npd['INT'] = npd['DF1'] * npd['LF1']

            # Zscores of lags and changes
            npd['ZDF1'] = (npd['DF1'] - npd['DF1'].mean()) / npd['DF1'].std()
            npd['ZLF1'] = (npd['LF1'] - npd['LF1'].mean()) / npd['LF1'].std()
            npd['ZF1M'] = (npd['F1M'] - npd['F1M'].mean()) / npd['F1M'].std()

            # Time weights
            npd['TimeWeights'] = np.exp(((npd.index - npd.index[len(npd.index) - 1])
                                         .days / time_weight_param / 365.0))

            # Grid
            x = np.linspace(level_grid_min, level_grid_max, 20)
            y = np.linspace(change_grid_min, change_grid_max, 20)
            xv, yv = np.meshgrid(x, y)
            out_grid = [xv, yv]

            # nonparametric plot data (nppd)
            nppd = pd.DataFrame()
            nppd['ZF1M'] = npd['ZF1M']
            nppd['ZDF1'] = npd['ZDF1']
            indVar = np.array(nppd)
            depVar_res = np.array(npd['residual_volatility'])
            depVar = np.array(npd['Volatility'])

            # Run nonparametric analysis
            bandwidth = 1
            predGrid = stats.npreg(depVar,
                                   indVar,
                                   out_grid,
                                   bandwidth,
                                   npd['TimeWeights'].values)
            bandwidth = 1.5
            predGrid_res = stats.npreg(depVar_res,
                                       indVar,
                                       out_grid,
                                       bandwidth,
                                       npd['TimeWeights'].values)

            # Get interpolation
            f = interpolate.interp2d(x, y, predGrid, kind='linear')
            f_res = interpolate.interp2d(x, y, predGrid_res, kind='linear')

            return f, f_res, out_grid, npd


def market_pca():

    tickers = ['VT', 'ITOT', 'IWM', 'SPY', 'QQQ',
               'VGK', 'EEM', 'EWJ', 'AAXJ', 'EWY', 'EWZ', 'FXI',
               'HYG', 'USO', 'GLD', 'UUP', 'FXY', 'TLT',
               'XLE', 'XLY', 'XLB', 'XLI', 'XLP', 'XLU', 'XLK', 'XLV', 'XLF']

    start_date = dt.datetime(2005, 1, 1)

    raw_data = pdata.get_data_yahoo(tickers, start_date)
    data = raw_data['Adj Close']

    return_window_days = 5
    returns = data / data.shift(return_window_days) - 1
    returns_clean = returns[np.isfinite(returns).all(axis=1)]

    returns_clean_z = (returns_clean - returns_clean.mean(axis=0)) \
                      / returns_clean.std(axis=0)

    fit_start_date = dt.datetime(2014, 1, 1)
    returns_fit = returns_clean_z[returns_clean_z.index > fit_start_date]
    p = PCA(whiten=True)
    x = p.fit_transform(returns_fit)
    x = pd.DataFrame(index=returns_fit.index, data=x)

    # plt.plot(x[[0, 1, 2]].cumsum())

    w = pd.DataFrame(columns=returns.columns,
                     data=p.components_)

    # factor_index = 2
    # pos = np.arange(len(tickers)) + 0.5
    # plt.figure(1)
    # w0 = w.iloc[factor_index].sort_values()
    # plt.barh(pos, w0, align='center')
    # labels = tuple(w0.index)
    # plt.yticks(pos, labels)


def plot_graph(settings=None, macro_data_z=None, negate_fields=None):

    symbols = np.array(settings.data_fieldnames).T
    graph_data = macro_data_z[macro_data_z.index > settings.common_start_date
                             ][settings.data_fields].iloc[2:]
    if negate_fields is not None:
        graph_data[negate_fields] = -graph_data[negate_fields]

    graph_data = graph_data.rolling(window=3, center=False).sum()
    variation = graph_data.values.T

    ###############################################################################
    # Learn a graphical structure from the correlations
    edge_model = covariance.GraphLassoCV()

    # standardize the time series: using correlations rather than covariance
    # is more efficient for structure recovery
    X = variation.copy().T
    X /= X.std(axis=0)
    edge_model.fit(X)

    ###############################################################################
    # Cluster using affinity propagation

    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    n_labels = labels.max()

    for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(symbols[labels == i])))

    ###############################################################################
    # Find a low-dimension embedding for visualization: find the best position of
    # the nodes (the stocks) on a 2D plane
    from sklearn.decomposition import kernel_pca
    # node_position_model = manifold.LocallyLinearEmbedding(
    #     n_components=2, eigen_solver='dense', n_neighbors=8)
    # node_position_model = KernelPCA(kernel='rbf',
    #                                 fit_inverse_transform=True,
    #                                 gamma=10,
    #                                 n_components=2)
    node_position_model = manifold.SpectralEmbedding(n_components=2,
                                                     n_neighbors=6)

    # node_position_model = PCA(n_components=2)
    embedding = node_position_model.fit_transform(X.T).T
    # embedding = components[[0, 1]].values.T
    f1 = 0
    f2 = 1

    ###############################################################################
    # Visualization
    plt.figure(1, facecolor='w', figsize=(12, 6))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    # plt.axis('off')
    # ax.set_axis_bgcolor('k')

    # Display a graph of the partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[f1],
                embedding[f2],
                s=100 * d ** 2,
                c=labels,
                cmap=plt.cm.coolwarm)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    segments = [[embedding[[f1, f2], start], embedding[[f1, f2], stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0,
                        cmap=plt.cm.coolwarm,
                        norm=plt.Normalize(0, .7 * np.sqrt(values.max())))
    lc.set_array(np.sqrt(values))
    lc.set_linewidths(15 * np.sqrt(values))
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    label_offset = 0.002

    for index, (name, label, (f_1, f_2)) in enumerate(
            zip(symbols, labels, embedding.T)):

        if f1 == 0:
            x = f_1
        if f1 == 1:
            x = f_2

        if f2 == 0:
            y = f_1
        if f2 == 1:
            y = f_2

        dx = x - embedding[f1]
        dx[index] = 1
        dy = y - embedding[f2]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x += label_offset
        else:
            horizontalalignment = 'right'
            x -= label_offset
        if this_dy > 0:
            verticalalignment = 'bottom'
            y += label_offset
        else:
            verticalalignment = 'top'
            y -= label_offset
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.spectral(label / float(n_labels)),
                           alpha=.6))

    plt.xlim(embedding[f1].min() - .15 * embedding[f1].ptp(),
             embedding[f1].max() + .10 * embedding[f1].ptp(),)
    plt.ylim(embedding[f2].min() - .03 * embedding[f2].ptp(),
             embedding[f2].max() + .03 * embedding[f2].ptp())
    plt.show()

    plt.savefig('figures/macro_graph.png',
                facecolor='w',
                edgecolor='w',
                transparent=True)