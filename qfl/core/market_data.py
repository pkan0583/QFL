import os
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import BDay
import numpy as np

import logging

import qfl.core.constants as constants
import qfl.core.calcs as calcs
import qfl.utilities.basic_utilities as utils
from qfl.core.database_interface import DatabaseInterface as db
from qfl.core.database_interface import DatabaseUtilities as dbutils

from scipy.interpolate import interp1d

market_data_date = utils.workday(dt.datetime.today(), num_days=-1)
history_default_start_date = dt.datetime(2000, 1, 1)
option_delta_grid = [1] + range(5, 95, 5) + [99]

db.initialize()


"""
-------------------------------------------------------------------------------
UTILITIES
-------------------------------------------------------------------------------
"""


def get_futures_calendar_name(futures_series=None):
    s = db.get_futures_series(futures_series=futures_series)
    exchange_code = s.iloc[0]['exchange']
    calendar_name = utils.DateUtils.exchange_calendar_map[exchange_code]
    return calendar_name

"""
-------------------------------------------------------------------------------
SECURITIES UNIVERSES
-------------------------------------------------------------------------------
"""


def get_etf_vol_universe():

    # s = "select ticker, security_sub_type," \
    #     "avg(volume * last_price) as px_volume " \
    #     "from equity_prices_view where date >= '{0}'".format(from_date.date())
    # s += " group by ticker, security_sub_type "
    # d = db.read_sql(s)
    #
    # d = d[d['px_volume'] > volume_threshold]
    # d = d[d['security_sub_type'] == 'ETP']
    #
    # # Need a data quality "liquidity" test
    # tickers = [str(t) for t in d['ticker'].values.tolist()]

    universe =  ['DIA',
                 'EEM',
                 'EFA',
                 'EMB',
                 'EWJ',
                 'EWW',
                 'EWY',
                 'EWZ',
                 'EZU',
                 'FXI',
                 'GDX',
                 'GLD',
                 'HEDJ',
                 'HYG',
                 'IBB',
                 'IEF',
                 'IVV',
                 'IWF',
                 'IWM',
                 'IYR',
                 'JNK',
                 'JNUG',
                 'KBE',
                 'KRE',
                 'LQD',
                 'OIH',
                 'QQQ',
                 'RSX',
                 'SLV',
                 'SMH',
                 'SPY',
                 'SQQQ',
                 'TIP',
                 'TLT',
                 'TQQQ',
                 'TZA',
                 'USO',
                 'VGK',
                 'VWO',
                 'XLB',
                 'XLE',
                 'XLF',
                 'XLI',
                 'XLK',
                 'XLP',
                 'XLU',
                 'XLV',
                 'XLY',
                 'XOP',
                 'XRT']
    return universe


def get_etf_universe():

    equity_etfs = ['SPY', 'IWM', 'QQQ', 'VGK', 'EFA', 'EWJ', 'EWY', 'EWZ',
                   'EWU', 'FXI', 'EEM', 'VWO', 'RSX', 'DIA', 'FEZ', 'HEDJ',
                   'EWH', 'EWW']
    sector_etfs = ['XLK', 'XLE', 'XLY', 'XLP', 'XLU', 'XLV', 'XLB', 'XLI']
    industry_etfs = ['XOP', 'XRE', 'GDX', 'AMLP', 'IYR', 'KRE', 'OIH', 'XRT',
                     'KBE', 'SMH', 'XHB', 'AMJ']
    macro_etfs = ['USO', 'GLD', 'SLV', 'UUP', 'UNG', 'DBC', 'DBA']
    fi_etfs = ['TLT', 'TBT', 'HYG', 'JNK', 'LQD', 'AGG', 'IEF', 'BND', 'TIP',
               'SHV', 'EMB', 'MUB', 'BKLN', 'CIU', 'IEI']

    # RIGHT NOW THESE ARE ALL US...:
    etf_tickers = equity_etfs + sector_etfs + industry_etfs + macro_etfs + fi_etfs
    etf_exch_codes = ['US'] * len(etf_tickers)

    return etf_tickers, etf_exch_codes


def get_futures_contracts_by_series(futures_series=None,
                                    start_date=history_default_start_date,
                                    end_date=None):

    s = "select * from futures_contracts where series_id in " \
        " (select id from futures_series where series in {0}".format(
            dbutils.format_for_query(futures_series)) + ")"
    s += " and maturity_date >= '" + start_date.__str__() + "'"
    if end_date is not None:
        s += " and maturity_date <= '" + end_date.__str__() + "'"
    s += " order by maturity_date asc"
    data = db.read_sql(s)

    data['maturity_date'] = pd.to_datetime(data['maturity_date'])

    return data


def get_futures_generic_contract_map(futures_series=None,
                                     futures_prices=None,
                                     price_field='settle_price',
                                     start_date=history_default_start_date):

    futures_contracts = get_futures_contracts_by_series(
        futures_series=futures_series)
    futures_contracts = futures_contracts[['ticker', 'id', 'maturity_date']]
    futures_contracts = futures_contracts.rename(
        columns={'id': 'futures_contract_id'})

    if futures_prices is None:
        futures_prices = get_generic_futures_prices_by_series(
            futures_series=futures_series,
            start_date=start_date
        )

    # check for improperly formatted futures prices
    if 'volume' in futures_prices.columns:
        futures_prices = (futures_prices[price_field]).unstack(level='ticker')

    # objective is to assign each day / generic future to its actual contract
    dates = futures_prices.index.get_level_values('date')
    futures_contract_map = pd.DataFrame(index=futures_prices.index,
                                        columns=futures_prices.columns)
    for i in range(1, len(futures_contracts)):
        maturity_date = futures_contracts.iloc[i]['maturity_date']
        prev_maturity_date = futures_contracts.iloc[i - 1]['maturity_date']
        ind = futures_prices.index[(dates >= prev_maturity_date)
                                   & (dates < maturity_date)]
        if len(ind) == 0:
            continue
        k = 0
        for col in futures_contract_map.columns:
            if i + k < len(futures_contracts):
                futures_contract_map.loc[ind, col] = \
                futures_contracts.iloc[i + k]['ticker']
                k += 1

    return futures_contract_map, futures_contracts

"""
-------------------------------------------------------------------------------
TIME SERIES DATA
-------------------------------------------------------------------------------
"""


def _get_time_series_data(s=None,
                          start_date=market_data_date,
                          end_date=None,
                          index_fields=None,
                          date_fields=['date']):

    s += " and date >= '" + start_date.__str__() + "'"
    if end_date is not None:
        s += " and date <= '" + end_date.__str__() + "'"
    s += " order by date asc"
    data = db.read_sql(s)

    for field in date_fields:
        data[field] = pd.to_datetime(data[field])
    if index_fields is not None:
        tmp = list()
        for field in index_fields:
            tmp.append(data[field])
        data.index = tmp
        for col in index_fields:
            del data[col]
    return data


def get_generic_index_prices(tickers=None,
                             start_date=market_data_date,
                             end_date=None):
    s = "select * from generic_index_prices_view " \
        " where ticker in {0}".format(dbutils.format_for_query(tickers))
    prices = _get_time_series_data(s=s,
                                   start_date=start_date,
                                   end_date=end_date,
                                   index_fields=['date', 'ticker'])
    return prices

def get_equity_tick_realized_volatility(tickers=None,
                                        fields=None,
                                        start_date=market_data_date,
                                        end_date=None):

    if fields is None:
        fields = ['tick_rv_10d',
                  'tick_rv_20d',
                  'tick_rv_60d',
                  'tick_rv_120d',
                  'tick_rv_252d',
                  'rv_10d',
                  'rv_20d',
                  'rv_60d',
                  'rv_120d',
                  'rv_252d']

    if fields is not None:
        fs = ""
        if 'ticker' not in fields:
            fs += 'ticker, '
        if 'date' not in fields:
            fs += 'date, '
        for field in fields:
            fs += field + ','
        fs = fs[0:len(fs)-1]

        s = "select " + fs + " from staging_orats" \
                             " where ticker in {0}".format(
                            dbutils.format_for_query(tickers))
        data = _get_time_series_data(s=s,
                                     start_date=start_date,
                                     end_date=end_date,
                                     index_fields=['date', 'ticker'])
        return data


def get_equity_implied_volatility(tickers=None,
                                  fields=None,
                                  start_date=market_data_date,
                                  end_date=None):
    fs = "*"
    if fields is not None:
        fs = ""
        if 'ticker' not in fields:
            fs += 'ticker, '
        if 'date' not in fields:
            fs += 'date, '
        for field in fields:
            fs += field + ','
        fs = fs[0:len(fs)-1]

    s = "select " + fs + " from staging_orats" \
        " where ticker in {0}".format(dbutils.format_for_query(tickers))
    data = _get_time_series_data(s=s,
                                 start_date=start_date,
                                 end_date=end_date,
                                 index_fields=['date', 'ticker'])
    return data


def get_equity_prices(tickers=None,
                      price_field='adj_close',
                      start_date=market_data_date,
                      end_date=None):

    s = "select ticker, date, " + price_field + " from equity_prices_view" \
        " where ticker in {0}".format(dbutils.format_for_query(tickers))
    prices = _get_time_series_data(s=s,
                                   start_date=start_date,
                                   end_date=end_date,
                                   index_fields=['date', 'ticker'])

    return prices


def get_futures_prices_by_tickers(tickers=None,
                                  start_date=market_data_date,
                                  end_date=market_data_date):

    s = "select * from futures_prices_view" \
        " where ticker in {0}".format(dbutils.format_for_query(tickers))
    prices = _get_time_series_data(s=s,
                                   start_date=start_date,
                                   end_date=end_date,
                                   index_fields=['date', 'ticker'])
    return prices


def get_rolling_futures_returns_by_series(futures_series=None,
                                          start_date=market_data_date,
                                          end_date=None,
                                          price_field='settle_price',
                                          level_change=False,
                                          days_to_zero_around_roll=1):

    generic_futures_data = get_generic_futures_prices_by_series(
        futures_series=futures_series,
        start_date=start_date,
        end_date=end_date)[
        [price_field, 'days_to_maturity']] \
        .unstack('ticker')

    return calcs.compute_rolling_futures_returns(
        generic_futures_data=generic_futures_data,
        price_field=price_field,
        level_change=level_change,
        days_to_zero_around_roll=days_to_zero_around_roll)


def get_constant_maturity_futures_prices_by_series(futures_series=None,
                                                   start_date=market_data_date,
                                                   end_date=None):

    table_name = 'constant_maturity_futures_prices_view'

    s = "select * from " + table_name
    s += " where series in {0}".format(
            dbutils.format_for_query(futures_series))
    prices = _get_time_series_data(s=s,
                                   start_date=start_date,
                                   end_date=end_date,
                                   index_fields=['date',
                                                 'series',
                                                 'days_to_maturity'])
    return prices


def get_futures_prices_by_series(futures_series=None,
                                 start_date=market_data_date,
                                 end_date=None):

    table_name = 'futures_prices_view'

    s = "select * from " + table_name + " where series in {0}".format(
        dbutils.format_for_query(futures_series))
    prices = _get_time_series_data(s=s,
                                   start_date=start_date,
                                   end_date=end_date,
                                   index_fields=['date', 'ticker'])
    return prices


def get_generic_futures_prices_by_series(futures_series=None,
                                         start_date=market_data_date,
                                         end_date=None,
                                         mapped_view=True):

    if mapped_view:
        table_name = 'generic_futures_prices_mapped_view'
    else:
        table_name = 'generic_futures_prices_view'

    s = "select * from " + table_name + " where series in {0}".format(
        dbutils.format_for_query(futures_series))

    prices = _get_time_series_data(s=s,
                                   start_date=start_date,
                                   end_date=end_date,
                                   index_fields=['date', 'ticker'])

    return prices


def get_cftc_positioning_by_series(futures_series=None,
                                   start_date=market_data_date,
                                   end_date=None,
                                   cftc_financial=True,
                                   columns=None):

    table_name = 'staging_cftc_positioning_financial'
    if not cftc_financial:
        table_name = 'staging_cftc_positioning_commodity'

    if columns is None:
        columns = ['report_date_as_mm_dd_yyyy',
                   'open_interest_all',
                   'dealer_positions_long_all',
                   'dealer_positions_short_all',
                   'asset_mgr_positions_long_all',
                   'asset_mgr_positions_short_all',
                   'lev_money_positions_long_all',
                   'lev_money_positions_short_all']

    # Ensure that we get the most recent data
    one_week_prior = start_date - BDay(5)

    fs = db.get_futures_series(futures_series)
    if len(fs) == 0:
        return None

    cftc_code = str(fs['cftc_code'].values[0])

    s = "select {0}".format(columns)\
            .replace('[', '').replace(']', '').replace("'", '') + \
        " from " + table_name + \
        ' where cftc_contract_market_code in {0}'.format(
        dbutils.format_for_query([cftc_code]))

    s += " and report_date_as_mm_dd_yyyy >= '{0}'".format(one_week_prior)
    if end_date is not None:
        s += " and report_date_as_mm_dd_yyyy <= '{0}'".format(end_date)

    s += ' order by report_date_as_mm_dd_yyyy asc'
    data = db.read_sql(s)
    data = data.rename(columns={'report_date_as_mm_dd_yyyy': 'date'})

    return data


def get_futures_ivol_by_series(futures_series=None,
                               maturity_type=None,
                               start_date=history_default_start_date,
                               end_date=market_data_date,
                               option_type="call",
                               option_deltas=None):

    # Get series id
    series_id = db.get_futures_series(futures_series)['id'].values.tolist()

    # Handle maturity type choice
    if maturity_type == 'constant_maturity':
        table_name = "futures_ivol_constant_maturity_by_delta"
    elif maturity_type == 'futures_contract':
        table_name = "futures_ivol_fixed_maturity_by_delta"
    else:
        raise ValueError("maturity_type must be constant_maturity"
                         " or futures_contract")

    # TODO: move this to someplace central
    if option_type.lower() == "c":
        option_type = "call"
    if option_type.lower() == "p":
        option_type = "put"
    if not option_type.lower() in ["put", "call"]:
        raise ValueError("option_type must be put or call")

    where_str = " option_type = '{0}'".format(option_type)
    where_str += " and date >= '{0}'".format(start_date)
    where_str += " and date <= '{0}'".format(end_date)
    where_str += " and series_id in {0}".format(
        dbutils.format_for_query(series_id))

    # Case: specific delta requested
    if option_deltas is not None and option_type is not None:
        delta_str = list()
        for delta in option_deltas:
            if delta in option_delta_grid:
                if delta < 10:
                    delta_str.append('ivol_0' + str(delta) + 'd')
                else:
                    delta_str.append('ivol_' + str(delta) + 'd')
        s = " select series_id, date, days_to_maturity, {0}".format(delta_str)
        s += " from " + table_name
    else:
        s = " select * from " + table_name
    s += " where " + where_str + " order by date asc, days_to_maturity asc"
    data = db.read_sql(s)
    return data


def get_optionworks_staging_ivm(codes=None,
                                start_date=history_default_start_date,
                                end_date=dt.datetime.today()):

    s = "select * from staging_optionworks_ivm where code in {0}".format(
        dbutils.format_for_query(codes))

    data = _get_time_series_data(s=s,
                                 start_date=start_date,
                                 end_date=end_date,
                                 index_fields=['date', 'code'])
    return data


def get_optionworks_staging_ivs(codes=None,
                                start_date=history_default_start_date,
                                end_date=dt.datetime.today()):

    s = "select * from staging_optionworks_ivs where code in {0}".format(
        dbutils.format_for_query(codes))

    data = _get_time_series_data(s=s,
                                 start_date=start_date,
                                 end_date=end_date,
                                 index_fields=['date', 'code'])
    return data


"""
-------------------------------------------------------------------------------
VOLATILITY SURFACE MANAGER
-------------------------------------------------------------------------------
"""


class RealizedVolatilityManager(object):

    data=None

    def __init__(self):

        # Initialize data struture
        tickers = ['SPY']
        start_date = utils.workday(num_days=-2)
        self.data = get_equity_prices(tickers=tickers,
                                      start_date=start_date)


class VolatilitySurfaceManager(object):

    data=None

    ivol_fields = [
        'ticker', 'date',
        'iv_1m', 'iv_2m', 'iv_3m',
        'iv_1mc', 'iv_2mc', 'iv_3mc', 'iv_4mc',
        'days_to_maturity_1mc', 'days_to_maturity_2mc',
        'days_to_maturity_3mc', 'days_to_maturity_4mc',
        'skew', 'curvature', 'skew_inf', 'curvature_inf'
    ]

    def __init__(self):

        # Initialize data struture
        tickers = ['SPY']
        start_date = utils.workday(num_days=-2)
        self.data = get_equity_implied_volatility(tickers=tickers,
                                                  fields=self.ivol_fields,
                                                  start_date=start_date)

    def load_data(self,
                  tickers=None,
                  start_date=history_default_start_date,
                  end_date=dt.datetime.today()):

        data_1 = get_equity_implied_volatility(tickers=tickers,
                                               fields=self.ivol_fields,
                                               start_date=start_date,
                                               end_date=end_date)

        self.data = self.data.append(data_1).drop_duplicates()

    def clean_data(self):

        # # Cleaning implied volatility data
        # clean_ivol = pd.DataFrame(index=ivol.index, columns=ivol.columns)
        #
        # clean_ivol['iv_2m'], normal_tests_2m = calcs.clean_implied_vol_data(
        #     tickers=tickers,
        #     stock_prices=stock_prices,
        #     ivol=ivol['iv_2m'],
        #     ref_ivol_ticker='SPY',
        #     deg_f=5,
        #     res_com=5
        # )
        #
        # clean_ivol['iv_3m'], normal_tests_3m = calcs.clean_implied_vol_data(
        #     tickers=tickers,
        #     stock_prices=stock_prices,
        #     ivol=ivol['iv_3m'],
        #     ref_ivol_ticker='SPY',
        #     deg_f=5,
        #     res_com=5
        # )

        x=1

    def get_data(self,
                 tickers=None,
                 fields=None,
                 start_date=None,
                 end_date=None):

        data = self.data[
            self.data.index.get_level_values('ticker').isin(tickers)]

        if start_date is not None:
            data = data[data.index.get_level_values('date') >= start_date]

        if end_date is not None:
            data = data[data.index.get_level_values('date') <= end_date]

        if fields is not None:
            data = data[fields]

        return data

    def get_roll_schedule(self,
                          tickers=None,
                          start_date=history_default_start_date,
                          end_date=dt.datetime.today()):

        maturity_dates = dict()

        data = self.get_data(tickers=tickers,
                             start_date=start_date,
                             end_date=end_date)

        for ticker in tickers:

            tk_data = data[data.index.get_level_values('ticker') == ticker]
            days_to_maturity = tk_data[['days_to_maturity_1mc',
                                        'days_to_maturity_2mc',
                                        'days_to_maturity_3mc',
                                        'days_to_maturity_4mc']]

            # Pick a conservative interval, loop, find contracts
            interval = 10
            tmp_range = range(0, len(tk_data), interval)
            dates = data.index.get_level_values('date')
            if len(data) - 1 not in tmp_range:
                tmp_range += [len(data) - 1]
            maturity_dates[ticker] = list()
            for t in tmp_range:
                mat = utils.calendarday(
                    date=dates[t],
                    num_days=days_to_maturity.iloc[t]).tolist()
                maturity_dates[ticker].append(mat)
            maturity_dates[ticker] = np.unique(maturity_dates[ticker])

        return maturity_dates

    def get_surface_point(self,
                          data=None,
                          tickers=None,
                          call_delta=0.50,
                          tenor_in_days=21,
                          start_date=history_default_start_date,
                          end_date=dt.datetime.today()):

        """
        This interpolates a single point on the constant-maturity volatility
        surface in delta terms.
        :param tickers: list
        :param data: this is optional (DataFrame)
        :param call_delta:
        :param tenor_in_days:
        :param start_date:
        :param end_date:
        :return:
        """

        # current data source limitation
        raw_data_tenors = [21, 42, 63]

        if utils.is_iterable(tenor_in_days):
            tenor_in_days = np.clip(tenor_in_days, 21, 63)
        else:
            tenor_in_days = float(max(21, min(63, tenor_in_days)))

        # checking data format
        if call_delta > 1.0:
            call_delta /= 100.0

        if data is None:
            data = self.get_data(tickers=tickers,
                                 start_date=start_date,
                                 end_date=end_date)

        # We store data in this format:
        # IV = ATMIV * (1 + slope/1000 +
        #      (curve/1000 * (delta*100 - 50)/2) * delta*100-50))

        # We store the 1-month (21-day) skew/crv the 'inf' skew/crv
        # skew = w * 1m_skew + (1-w) * inf_skew
        # w = sqrt(21)/sqrt(252)
        w = (constants.trading_days_per_month / tenor_in_days) ** 0.5
        if utils.is_iterable(w):
            w = np.clip(w, 0, 1)
        else:
            if w > 1.0:
                w = 1.0

        skew = w * data['skew'] + (1-w) * data['skew_inf']
        curve = w * data['curvature'] + (1-w) * data['curvature_inf']

        tmp_int = interp1d(raw_data_tenors,
                           data[['iv_1m', 'iv_2m', 'iv_3m']] ** 2.0)

        # This is the ATM
        atm_iv = tmp_int(tenor_in_days) ** 0.5

        # Special case: raw_data_tenors was an iterable that matched the length
        # of the data, which means we want a different tenor for each row
        if utils.is_iterable(tenor_in_days):
            if len(tenor_in_days) == len(data):
                atm_iv = np.diag(atm_iv)

        # Now we adjust for the delta
        if call_delta == 0.5:
            iv = atm_iv
        else:
            iv = atm_iv * (1 + (call_delta * 100.0 - 50.0) *
                (skew / 1000.0 + (curve / 1000.0 *
                (call_delta * 100 - 50.0) / 2.0)))

        return iv







