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

market_data_date = utils.workday(dt.datetime.today(), num_days=-1)
history_default_start_date = dt.datetime(2000, 1, 1)
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
            dbutils.format_as_tuple_for_query(futures_series)) + ")"
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
        " where ticker in {0}".format(dbutils.format_as_tuple_for_query(tickers))
    prices = _get_time_series_data(s=s,
                                   start_date=start_date,
                                   end_date=end_date,
                                   index_fields=['date', 'ticker'])
    return prices


def get_equity_implied_volatility(tickers=None,
                                  fields=None,
                                  start_date=market_data_date,
                                  end_date=None):
    s = "select * from staging_orats" \
        " where ticker in {0}".format(dbutils.format_as_tuple_for_query(tickers))
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
        " where ticker in {0}".format(dbutils.format_as_tuple_for_query(tickers))
    prices = _get_time_series_data(s=s,
                                   start_date=start_date,
                                   end_date=end_date,
                                   index_fields=['date', 'ticker'])

    return prices


def get_futures_prices_by_tickers(tickers=None,
                                  start_date=market_data_date,
                                  end_date=market_data_date):

    s = "select * from futures_prices_view" \
        " where ticker in {0}".format(dbutils.format_as_tuple_for_query(tickers))
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
    s = "select * from constant_maturity_futures_prices_view " \
        " where series in {0}".format(
            dbutils.format_as_tuple_for_query(futures_series))
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
    s = "select * from futures_prices_view where series in {0}".format(
        dbutils.format_as_tuple_for_query(futures_series))
    prices = _get_time_series_data(s=s,
                                   start_date=start_date,
                                   end_date=end_date,
                                   index_fields=['date', 'ticker'])
    return prices


def get_generic_futures_prices_by_series(futures_series=None,
                                         start_date=market_data_date,
                                         end_date=None,
                                         include_contract_map=False):

    s = "select * from generic_futures_prices_view where series in {0}".format(
        dbutils.format_as_tuple_for_query(futures_series))

    prices = _get_time_series_data(s=s,
                                   start_date=start_date,
                                   end_date=end_date,
                                   index_fields=['date', 'ticker'])

    if include_contract_map:

        # Map to actual contract months
        contract_map, futures_contracts = get_futures_generic_contract_map(
            futures_series=futures_series,
            price_field='settle_price',
            start_date=start_date)

        contract_map = contract_map.stack('ticker')

        contract_map.name = 'contract_ticker'

        prices = prices.join(contract_map)

        futures_contracts = futures_contracts.rename(
            columns={'ticker': 'contract_ticker'})

        # Map to expiration dates so we can calculate time to roll
        prices = pd.merge(left=futures_contracts,
                          right=prices.reset_index(),
                          on='contract_ticker')
        prices.index = [prices['date'], prices['ticker']]
        del prices['ticker']
        del prices['date']

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
                   'lev_Money_positions_long_all',
                   'lev_Money_positions_short_all']

    # Ensure that we get the most recent data
    one_week_prior = start_date - BDay(5)

    fs = db.get_futures_series(futures_series)
    if len(fs) == 0:
        return None

    cftc_code = str(fs['cftc_code'].values[0])

    s = "select {0}".format(columns)\
            .replace('[', '').replace(']', '').replace("'", '') + \
        " from " + table_name + \
        ' where "CFTC_Contract_Market_Code" in {0}'.format(
        dbutils.format_as_tuple_for_query([cftc_code]))

    s += " and report_date_as_mm_dd_yyyy >= '{0}'".format(one_week_prior)
    if end_date is not None:
        s += " and report_date_as_mm_dd_yyyy <= '{0}'".format(end_date)

    s += ' order by report_date_as_mm_dd_yyyy asc'
    data = db.read_sql(s)
    data = data.rename(columns={'report_date_as_mm_dd_yyyy': 'date'})

    return data


def get_optionworks_staging_ivm(codes=None,
                                start_date=history_default_start_date,
                                end_date=dt.datetime.today()):

    s = "select * from staging_optionworks_ivm where code in {0}".format(
        dbutils.format_as_tuple_for_query(codes))

    data = _get_time_series_data(s=s,
                                 start_date=start_date,
                                 end_date=end_date,
                                 index_fields=['date', 'code'])
    return data


def get_optionworks_staging_ivs(codes=None,
                                start_date=history_default_start_date,
                                end_date=dt.datetime.today()):

    s = "select * from staging_optionworks_ivs where code in {0}".format(
        dbutils.format_as_tuple_for_query(codes))

    data = _get_time_series_data(s=s,
                                 start_date=start_date,
                                 end_date=end_date,
                                 index_fields=['date', 'code'])
    return data