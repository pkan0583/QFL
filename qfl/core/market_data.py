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

market_data_date = utils.workday(dt.datetime.today().date(), num_days=-1)
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
MARKET DATA MANAGER
-------------------------------------------------------------------------------
"""


class MarketDataManager(object):

    # Objective here is to provide generic loading and storage with caching

    data_categories = [
        'equity_price',
        'equity_implied_volatility',
        'futures_price',
        'generic_futures_price',
        'futures_implied_volatility',
        'yield_curve',
        'currency',
        'equity_index_price'
    ]

    cached_data = dict()
    cache_manager = dict()
    vsm = None

    def __init__(self):

        cols = ['ticker', 'start_date', 'end_date']
        for data_category in self.data_categories:
            self.cache_manager[data_category] = pd.DataFrame(columns=cols)

        vsm = VolatilitySurfaceManager()

    def get_data(self,
                 data_category=None,
                 tickers=None,
                 start_date=history_default_start_date,
                 end_date=market_data_date,
                 **kwargs):

        if 1 == 1:
            pass
        elif data_category == 'equity_implied_volatility':

            moneyness = kwargs.get('moneyness', None)
            delta = kwargs.get('moneyness', None)
            tenor_in_days = kwargs.get('tenor_in_days', None)
            maturity_date = kwargs.get('maturity_date', None)
            fields = kwargs.get('fields', None)

            if delta is not None and tenor_in_days is not None:
                req_data = self.vsm.get_surface_point_by_delta(
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    call_delta=delta
                )
            else:
                req_data = get_equity_implied_volatility(tickers=tickers,
                                                         fields=fields,
                                                         start_date=start_date,
                                                         end_date=end_date)

            return req_data

    def _load_data(self,
                   data_category,
                   tickers=None,
                   start_date=history_default_start_date,
                   end_date=market_data_date,
                   **kwargs):

        if data_category == 'equity_price':

            price_field = kwargs.get('price_field', 'adj_close')
            req_data = get_equity_prices(tickers=tickers,
                                         start_date=start_date,
                                         end_date=end_date,
                                         price_field=price_field)

        elif data_category == 'equity_implied_volatility':

            self.vsm.load_data(tickers=tickers,
                               start_date=start_date,
                               end_date=end_date)

        elif data_category == 'equity_index_price':

            price_field = kwargs.get('price_field', 'last_price')
            req_data = get_equity_index_prices(tickers=tickers,
                                               start_date=start_date,
                                               end_date=end_date,
                                               price_field=price_field)

        elif data_category == 'futures_price':

            by_series = kwargs.get('by_series', True)
            if by_series:
                req_data = get_futures_prices_by_series(futures_series=tickers,
                                                        start_date=start_date,
                                                        end_date=end_date)
            else:
                req_data = get_futures_prices_by_tickers(tickers=tickers,
                                                         start_date=start_date,
                                                         end_date=end_date)

        elif data_category == 'generic_futures_price':

            constant_maturity = False

            if constant_maturity:
                req_data = get_constant_maturity_futures_prices_by_series(
                    futures_series=tickers,
                    start_date=start_date,
                    end_date=end_date,

                )
            else:
                req_data = get_generic_futures_prices_by_series(
                        futures_series=tickers,
                        start_date=start_date,
                        end_date=end_date)




        # Initialize the dataset if necessary
        if data_category not in self.cached_data:
            self.cached_data[data_category] = req_data

        # Drop duplicate rows and sort
        df = self.cached_data[data_category].append(req_data)
        df = df.groupby(level=df.index.names).last().sort_index()
        self.cached_data[data_category] = df

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


def get_equity_index_prices(tickers=None,
                            price_field='last_price',
                            start_date=market_data_date,
                            end_date=None):
    s = "select ticker, date, " + price_field \
        + " from equity_index_prices_view" \
        + " where ticker in {0}".format(dbutils.format_for_query(tickers))
    prices = _get_time_series_data(s=s,
                                   start_date=start_date,
                                   end_date=end_date,
                                   index_fields=['date', 'ticker'])
    return prices


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

        dates = self.data.index.get_level_values('date')
        for i in range(1, 5):
            col1 = 'maturity_date_' + str(i) + 'mc'
            col2 = 'days_to_maturity_' + str(i) + 'mc'
            self.data[col1] = \
                utils.calendarday(date=dates, num_days=self.data[col2])

        self.data = self.data.sort_index()

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

        pass

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

    def get_surface_point_by_moneyness(
            self,
            data=None,
            tickers=None,
            moneyness=[1.0],
            tenor_in_days=21,
            start_date=history_default_start_date,
            end_date=dt.datetime.today()):

        pass

    def get_fixed_maturity_vol_by_strike(
            self,
            ticker=None,
            strikes=None,
            contract_month_number=1,
            tenor_in_days=None,
            start_date=history_default_start_date,
            end_date=dt.datetime.today(),
            strike_as_moneyness=False):

        """
        Get a time series of implied volatilities for a fixed-maturity
        Maturity is a time series
        :param ticker:
        :param strikes:
        :param contract_month_number:
        :param start_date:
        :param end_date:
        :param strike_as_moneyness:
        :return:
        """

        delta_grid = np.append([0.001, 0.01],
                               np.append(np.arange(0.05, 1.0, 0.05),
                                         [0.99, 0.999]))

        if tenor_in_days is None:
            fm_data = self.get_data(tickers=[ticker],
                                    start_date=start_date,
                                    end_date=end_date)
            fm_data = fm_data[['iv_1mc', 'iv_2mc', 'iv_3m', 'iv_4mc',
                           'days_to_maturity_1mc', 'days_to_maturity_2mc',
                           'days_to_maturity_3mc', 'days_to_maturity_4mc',
                           'skew', 'curvature', 'skew_inf', 'curvature_inf']]
            tenor_string = 'days_to_maturity_' + str(
                contract_month_number) + 'mc'
            tenor_in_days = fm_data[tenor_string]\
                .reset_index(level='ticker', drop=True)

        fm_vols = self.get_surface_point_by_delta(
            tickers=[ticker],
            call_delta=delta_grid.tolist(),
            tenor_in_days=tenor_in_days,
            start_date=start_date,
            end_date=end_date).reset_index(level='ticker', drop=True)

        fm_m = calcs.black_scholes_moneyness_from_delta(
            call_delta=delta_grid.tolist(),
            tenor_in_days=tenor_in_days,
            ivol=fm_vols / 100.0,
            risk_free=0,
            div_yield=0)

        sp = get_equity_prices(tickers=[ticker], start_date=start_date) \
            ['adj_close'] \
            .reset_index(level='ticker', drop=True)

        if strike_as_moneyness:
            strike_money = strikes
        else:
            strike_data = pd.DataFrame(index=sp.index, columns=strikes)
            for strike in strikes:
                strike_data[strike] = strike / sp
            strike_money = pd.DataFrame(index=fm_m.index, columns=strikes,
                                        data=strike_data)

        interp_vols = pd.DataFrame(index=fm_vols.index, columns=strikes)
        min_m_vol = fm_vols[delta_grid.min()]
        max_m_vol = fm_vols[delta_grid.max()]

        for t in range(0, len(fm_vols.index) - 1):
            date = fm_vols.index[t]
            tmp = interp1d(x=fm_m.loc[date], y=fm_vols.loc[date],
                           bounds_error=False)
            interp_vols.loc[date] = tmp(strike_money.loc[date])
            # hi_ind = interp_vols.loc[date].index[
            #     strike_money.iloc[t] < fm_m.loc[date].min()]
            # lo_ind = interp_vols.loc[date].index[
            #     strike_money.iloc[t] > fm_m.loc[date].max()]
            # interp_vols.loc[date, lo_ind] = min_m_vol.loc[date]
            # interp_vols.loc[date, hi_ind] = max_m_vol.loc[date]

        for strike in strikes:
            hi_ind = interp_vols[strike].index[
                strike_money[strike] < fm_m.min(axis=1)]
            lo_ind = interp_vols[strike].index[
                strike_money[strike] > fm_m.max(axis=1)]
            interp_vols.loc[lo_ind, strike] = min_m_vol.loc[lo_ind]
            interp_vols.loc[hi_ind, strike] = max_m_vol.loc[hi_ind]

        return interp_vols

    def get_surface_point_by_delta(self,
                                   data=None,
                                   tickers=None,
                                   call_delta=[0.50],
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

        if not utils.is_iterable(call_delta):
            call_delta = [call_delta]

        # current data source limitation
        raw_data_tenors = [21, 42, 63]

        if utils.is_iterable(tenor_in_days):
            tenor_in_days = np.clip(tenor_in_days, 21, 63)
        else:
            tenor_in_days = float(max(21, min(63, tenor_in_days)))

        # checking data format
        if call_delta[0] > 1.0:
            call_delta = [cd / 100.0 for cd in call_delta]

        if data is None:
            data = self.get_data(tickers=tickers,
                                 start_date=start_date,
                                 end_date=end_date)

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

        # Special case: tenor_in_days was an iterable that matched the length
        # of the data, which means we want a different tenor for each row
        if utils.is_iterable(tenor_in_days):
            if len(tenor_in_days) == len(data):
                atm_iv = np.diag(atm_iv)

        # Now we adjust for delta
        iv = pd.DataFrame(index=data.index, columns=call_delta)
        for delta in call_delta:
            if delta == 0.5:
                iv[delta] = atm_iv
            else:
                cd = (delta * 100.0 - 50.0)
                iv[delta] = atm_iv * (1 + cd *
                    (skew / 1000.0 + (curve / 1000.0 * cd / 2.0)))

        # Return series if only a single delta request
        if len(call_delta) == 1:
            iv = iv[call_delta[0]]

        return iv







