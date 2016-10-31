"""
This module should ideally be ONLY tasked with the automation and triggering of
ETL tasks and should rely on other modules to actually implement those tasks?
EG, we shouldn't need references here to YahooApi...
"""
import datetime as dt
import logging
import os
import time

import numpy as np
import odo
import pandas as pd
from pandas.tseries.offsets import BDay

import qfl.core.calcs as calcs
import qfl.core.constants as constants
import qfl.core.market_data as md
import qfl.etl.data_interfaces as qfl_data
import qfl.utilities.basic_utilities as utils
from qfl.core.database_interface import DatabaseInterface as db, \
                                        DatabaseUtilities as dbutils
from qfl.etl.data_interfaces import YahooApi, QuandlApi, FigiApi, DataScraper
import qfl.models.model_data_manager as mm

# Default start date for historical data ingests
default_start_date = dt.datetime(1990, 1, 1)
default_equity_indices = ['SPX', 'NDX', 'UKX']
db.initialize()

# US market close in UTC time
us_market_close_utc = dt.time(20, 15)


"""
-------------------------------------------------------------------------------
ETL UTILITIES
-------------------------------------------------------------------------------
"""


def _get_execution_date(date=None, prev_day=True, **kwargs):
    airflow_date = kwargs.get('execution_date', None)
    if airflow_date is not None:
        date = airflow_date
    elif date is None:
        date = dt.datetime.today()
    if prev_day:
        date = utils.workday(date, num_days=-1)
    return date


def _is_after_close(offset_hours=0, offset_minutes=0):

    todays_market_close = pd.to_datetime(dt.date.today()) + dt.timedelta(
        hours=us_market_close_utc.hour,
        minutes=us_market_close_utc.minute)

    if dt.datetime.utcnow() > todays_market_close \
            + dt.timedelta(hours=offset_hours, minutes=offset_minutes):
        return True
    else:
        return False


"""
-------------------------------------------------------------------------------
DATA INGEST: ENTRY POINTS
-------------------------------------------------------------------------------
"""


class DataIngest(object):

    name = ""

    @classmethod
    def launch(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def ingest_data(cls,
                    start_date=None,
                    end_date=None,
                    data_source_name=None):
        raise NotImplementedError

    @classmethod
    def check_data_ingest(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_start_and_end_dates(cls, **kwargs):
        raise NotImplementedError


class StandardDataIngest(DataIngest):

    name = ""
    expected_datestamp = None
    table_name = None
    check_non_null_column_name = None
    pass_rows_threshold = None
    data_source_name = None
    prev_day = False

    @classmethod
    def launch(cls, date=None, **kwargs):

        logging.info("Starting " + cls.name + " ... ")

        start_date, end_date = cls\
            .get_start_and_end_dates(date, **kwargs)

        data_source_name = kwargs.get('data_source_name',
                                      cls.data_source_name)
        cls.ingest_data(start_date, end_date, data_source_name, **kwargs)

        if cls.pass_rows_threshold > 0:
            success = cls.check_data_ingest(**kwargs)
        else:
            success = True

        if not success:
            raise RuntimeError('Data checks failed for '
                               + cls.name + '...')
        else:
            logging.info(cls.name + " success!")

    @classmethod
    def ingest_data(cls,
                    start_date=None,
                    end_date=None,
                    data_source_name=None,
                    **kwargs):
        raise NotImplementedError

    @classmethod
    def check_data_ingest(cls, **kwargs):

        logging.info("Checking " + cls.name + " ... ")

        table_name = cls.table_name
        pass_rows_threshold = cls.pass_rows_threshold
        where_str = " date = '{0}'".format(
            cls.expected_datestamp)
        where_str += " and " + cls.check_non_null_column_name \
                     + " is not null "
        test_data = db.get_data(table_name, where_str=where_str)
        success = False
        if len(test_data) > pass_rows_threshold:
            success = True
        return success

    @classmethod
    def get_start_and_end_dates(cls, date=None, **kwargs):
        date = _get_execution_date(date,
                                   prev_day=cls.prev_day,
                                   **kwargs)
        start_date = date
        end_date = date
        cls.expected_datestamp = date
        return start_date, end_date


class DailyGenericIndexPriceIngest(StandardDataIngest):

    name = "Daily Generic Index Price Ingest"
    expected_datestamp = None
    table_name = 'generic_index_prices'
    check_non_null_column_name = 'last_price'
    pass_rows_threshold = 2
    data_source_name = 'multiple'
    prev_day = True

    @classmethod
    def ingest_data(cls,
                    start_date=None,
                    end_date_=None,
                    data_source_name='multiple',
                    **kwargs):

        ingest_historical_volatility_index_prices(start_date=start_date)


class DailyEquityIndexPriceIngest(StandardDataIngest):

    name = "Daily Equity Index Price Ingest"
    expected_datestamp = None
    table_name = 'equity_index_prices'
    check_non_null_column_name = 'last_price'
    pass_rows_threshold = 10
    data_source_name = 'yahoo'
    prev_day = False

    @classmethod
    def ingest_data(cls,
                    start_date=None,
                    end_date_=None,
                    data_source_name='yahoo',
                    **kwargs):

        ingest_historical_index_prices(start_date=start_date,
                                       end_date=end_date_,
                                       data_source_name=data_source_name)


class DailyEquityPriceIngest(StandardDataIngest):

    name = "Daily Equity Price Ingest"
    expected_datestamp = None
    table_name = 'equity_prices'
    check_non_null_column_name = 'last_price'
    pass_rows_threshold = 3000
    data_source_name = 'yahoo'
    prev_day = False

    @classmethod
    def ingest_data(cls,
                    start_date=None,
                    end_date_=None,
                    data_source_name='yahoo',
                    **kwargs):
        # Prep
        equities, equity_prices_table, ids, equity_tickers, rows = \
            _prep_equity_price_ingest(ids=None)

        update_equity_prices(ids=ids,
                             data_source_name=data_source_name,
                             date=start_date)


class DailyGenericFuturesPriceIngest(StandardDataIngest):

    name = "Daily Generic Futures Price Ingest"
    expected_datestamp = None
    table_name = 'generic_futures_prices'
    check_non_null_column_name = 'settle_price'
    pass_rows_threshold = 10
    data_source_name = 'quandl'
    prev_day = True

    @classmethod
    def ingest_data(cls,
                    start_date=None,
                    end_date_=None,
                    data_source_name=None,
                    **kwargs):
        if data_source_name is None:
            data_source_name = cls.data_source_name
        update_generic_futures_prices(start_date=start_date,
                                      end_date=end_date_,
                                      data_source_name=data_source_name)


class DailyFuturesPriceIngest(StandardDataIngest):

    name = "Daily Futures Price Ingest"
    expected_datestamp = None
    table_name = 'futures_prices'
    check_non_null_column_name = 'settle_price'
    pass_rows_threshold = 10
    data_source_name = 'quandl'
    prev_day = True

    @classmethod
    def ingest_data(cls,
                    start_date=None,
                    end_date_=None,
                    data_source_name=None,
                    **kwargs):
        if data_source_name is None:
            data_source_name = cls.data_source_name
        update_futures_prices(start_date, end_date_, data_source_name)


class DailyOratsIngest(StandardDataIngest):

    name = "Daily ORATS Implied Volatility Data Ingest"
    expected_datestamp = None
    table_name = 'staging_orats'
    check_non_null_column_name = 'iv_1m'
    pass_rows_threshold = 3000
    data_source_name = 'quandl'
    prev_day = True

    @classmethod
    def ingest_data(cls,
                    start_date=None,
                    end_date_=None,
                    data_source_name=None,
                    **kwargs):
        if data_source_name is None:
            data_source_name = cls.data_source_name
        ingest_historical_orats_data(full_history=False)
        # ingest_historical_orats_data_from_api(start_date=start_date)


class DailyOptionWorksIngest(StandardDataIngest):

    name = "Daily OptionWorks Implied Volatility Ingest"
    expected_datestamp = None
    table_name = 'staging_optionworks_ivm'
    check_non_null_column_name = 'date'
    pass_rows_threshold = 1000
    data_source_name = 'quandl'
    prev_day = True

    @classmethod
    def ingest_data(cls,
                    start_date=None,
                    end_date_=None,
                    data_source_name=None,
                    **kwargs):
        if data_source_name is None:
            data_source_name = cls.data_source_name
        ingest_optionworks_data(full_history=False)
        process_all_optionworks_data(start_date=start_date)


class WeeklyCftcCommodityIngest(StandardDataIngest):

    name = "Weekly CFTC Commodity Positioning Data Ingest"
    expected_datestamp = None
    table_name = 'staging_cftc_positioning_commodity'
    check_non_null_column_name = 'CFTC_Contract_Market_Code'
    pass_rows_threshold = 0
    data_source_name = 'cftc'
    prev_day = False

    @classmethod
    def ingest_data(cls,
                    start_date=None,
                    end_date_=None,
                    data_source_name=None,
                    **kwargs):
        if data_source_name is None:
            data_source_name = cls.data_source_name
        ingest_cftc_positioning_data(category='commodity',
                                     full_history=False)


class WeeklyCftcFinancialsIngest(StandardDataIngest):

    name = "Weekly CFTC Financials Positioning Data Ingest"
    expected_datestamp = None
    table_name = 'staging_cftc_positioning_financial'
    check_non_null_column_name = 'CFTC_Contract_Market_Code'
    pass_rows_threshold = 0
    data_source_name = 'cftc'
    prev_day = False

    @classmethod
    def ingest_data(cls,
                    start_date=None,
                    end_date_=None,
                    data_source_name=None,
                    **kwargs):
        if data_source_name is None:
            data_source_name = cls.data_source_name
        ingest_cftc_positioning_data(category='financial',
                                     full_history=False)


def daily_orats_data_ingest(date=None, **kwargs):

    logging.info("starting daily ORATS ingest for " + str(date) + "...")
    ingest_historical_orats_data(full_history=False)
    logging.info("completed daily ORATS data ingest!")


def daily_equity_price_ingest(date=None, **kwargs):

    date = _get_execution_date(date, prev_day=False, **kwargs)

    logging.info("starting daily equity price ingest for "
                 + str(date) + "...")

    data_source_name = 'yahoo'

    # Prep
    equities, equity_prices_table, ids, equity_tickers, rows = \
        _prep_equity_price_ingest(ids=None)

    update_equity_prices(ids=ids,
                         data_source_name=data_source_name,
                         date=date)

    logging.info("completed daily equity price ingest!")


def historical_equity_price_ingest():

    date = utils.closest_business_day()
    data_source = 'yahoo'
    ingest_historical_equity_prices(ids=None,
                                    start_date=default_start_date,
                                    end_date=date,
                                    data_source=data_source)


def historical_dividends_ingest():

    ingest_historical_dividends(ids=None,
                                start_date=default_start_date,
                                end_date=dt.datetime.today(),
                                data_source='yahoo')


"""
-------------------------------------------------------------------------------
YAHOO OPTIONS
-------------------------------------------------------------------------------
"""


def _update_option_attrs(raw_data=None):

    options_table = db.get_table(table_name='equity_options')
    option_attributes = pd.DataFrame(
        raw_data.index.get_level_values(level='Symbol'))
    option_attributes = option_attributes.rename(
        columns={'Symbol': 'ticker'})
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

    return ticker_id_map, option_attributes['ticker']


def update_option_prices(ids=None,
                         data_source='yahoo'):
    for id_ in ids:
        update_option_prices_one(id_=id_,
                                 data_source=data_source)


def update_option_prices_one(id_=None,
                             data_source='yahoo'):

    # TODO: move the Yahoo-specific stuff out of here
    ticker = db.get_equity_tickers(ids=[id_])[0]

    if data_source == 'yahoo':

        # Get raw data
        tmp = YahooApi.retrieve_options_data(ticker)
        raw_data = tmp[0]

        # Update option universe
        ticker_id_map, tickers = _update_option_attrs(raw_data)

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

        ids = ticker_id_map.loc[tickers, 'id']
        option_prices = option_prices.reset_index()
        option_prices['id'] = ids.values

        db.execute_db_save(df=option_prices,
                           table=option_prices_table)

    else:
        raise NotImplementedError


"""
-------------------------------------------------------------------------------
EQUITY PRICES
-------------------------------------------------------------------------------
"""


def _prep_equity_price_ingest(ids=None,
                              equity_indices=None):

    if equity_indices is None:
        where_str = None
        if ids is not None:
            if len(ids) == 1:
                where_str = " id = {0}".format(ids[0])
            else:
                where_str = " id in {0}".format(tuple(ids))
        equities = db.get_data(table_name="equities", where_str=where_str)
    else:
        equities = db.get_equities(equity_indices=equity_indices)
    equities.index = equities['id']

    equity_prices_table = db.get_table(table_name='equity_prices')

    # Default is everything
    if ids is None:
        ids = equities.index.tolist()

    try:
        rows = equities.loc[ids].reset_index()
    except:
        rows = equities.loc[ids]

    equity_tickers = rows['ticker'].tolist()

    # handle potential unicode weirdness
    equity_tickers = [str(ticker) for ticker in equity_tickers]

    return equities, equity_prices_table, ids, equity_tickers, rows


def update_equity_prices(ids=None,
                         data_source_name='yahoo',
                         date=None,
                         batch_size=1):

    if date is None:
        date = utils.workday(num_days=-1)

    ingest_historical_equity_prices(ids=ids,
                                    start_date=date,
                                    data_source=data_source_name,
                                    batch=True,
                                    batch_size=batch_size)


def _ingest_historical_equity_prices(ids=None,
                                     start_date=default_start_date,
                                     end_date=dt.datetime.today(),
                                     data_source_name='yahoo',
                                     batch_size=20):
    """
    Internal method for ingesting
    :param ids:
    :param start_date:
    :param end_date:
    :param data_source_name:
    :param batch_size:
    :return:
    """
    # Prep
    if isinstance(ids, int):
        ids = [ids]
    equities, equity_prices_table, ids, equity_tickers, rows = \
        _prep_equity_price_ingest(ids=ids)

    if data_source_name == 'yahoo':

        # TODO: move this to a sensible place
        equity_tickers = [ticker.replace(".LN", ".L")
                          for ticker in equity_tickers]

        prices = YahooApi.retrieve_prices(equity_tickers,
                                          start_date,
                                          end_date)

        if isinstance(prices, pd.Panel):
            prices_df = prices.to_frame()
        else:
            prices_df = prices

        column_names = ['id', 'date', 'open_price', 'high_price', 'low_price',
                        'last_price', 'volume', 'adj_close']

        # Remove indices to prepare for database
        prices_df.index.names = ['date', 'ticker']
        prices_df = prices_df.reset_index()

        # Merge with ID's
        mapped_prices = pd.merge(left=prices_df,
                                 right=rows,
                                 on='ticker',
                                 how='inner')

    else:
        raise NotImplementedError

    # Map to database column structure
    equity_prices_data = pd.DataFrame(index=mapped_prices.index,
                                      columns=equity_prices_table.columns.keys())
    for i in range(0, len(column_names)):
        equity_prices_data[column_names[i]] = mapped_prices[column_names[i]]

    logging.info("archiving " + str(len(equity_prices_data)) + " equity prices...")

    equity_prices_data = equity_prices_data.reset_index()
    db.execute_db_save(df=equity_prices_data,
                       table=equity_prices_table,
                       time_series=True,
                       extra_careful=False)


def ingest_historical_equity_prices(ids=None,
                                    start_date=default_start_date,
                                    end_date=dt.datetime.today(),
                                    data_source='yahoo',
                                    batch=True,
                                    batch_size=5):

    if batch:

        num_batches = int(np.ceil(float(len(ids)) / batch_size))

        for i in range(0, num_batches):

            logging.info("batch " + str(i) + " out of " + str(num_batches))

            lrange = i * batch_size
            urange = np.min([(i + 1) * batch_size, len(ids)-1])

            logging.info("archiving batch " + str(i)
                         + " out of " + str(len(ids)))

            if len(ids) == 1 or batch_size == 1:
                batch_ids = [ids[lrange]]
            else:
                batch_ids = ids[lrange:urange]

            try:
                _ingest_historical_equity_prices(ids=batch_ids,
                                                 start_date=start_date,
                                                 end_date=end_date)
            except:
                logging.info(str(ids[i]) + " failed!")

    else:
        _ingest_historical_equity_prices(ids=ids,
                                         start_date=start_date,
                                         end_date=end_date,
                                         data_source_name=data_source,
                                         batch_size=batch_size)


"""
-------------------------------------------------------------------------------
EQUITY UNIVERSE
-------------------------------------------------------------------------------
"""


def add_equities_from_index_web():

    url_root = "http://www.dax-indices.com/MediaLibrary/Document/WeightingFiles/08/"
    url = url_root + "DAX_ICR.20160805.xls"
    file = urllib.urlretrieve(url, "data/dax.xls")
    tmp = pd.read_csv("data/dax.csv")

    import urllib
    url = "http://cfe.cboe.com/data/DailyVXFuturesEODValues/DownloadFS.aspx"
    file = urllib.urlretrieve(url, "data/vix_settle.csv")
    tmp = pd.read_csv("data/vix_settle.csv")


def add_etfs_to_universe(method='figi'):

    # Get our ETF universe, defined in code
    etf_list, etf_exch_codes = md.get_etf_universe()

    add_equities_to_security_universe(tickers=etf_list,
                                      exchange_codes=etf_exch_codes,
                                      method=method)


def add_equities_from_index(ticker=None,
                            exchange_code=None,
                            method='quandl'):

    # right now this uses quandl
    tickers = list()
    if method == 'quandl':
        if ticker not in QuandlApi.get_equity_index_universe():
            raise NotImplementedError
        tickers = QuandlApi.get_equity_universe(ticker)
    else:
        raise NotImplementedError

    # Make sure string format is normal
    tickers = [str(t) for t in tickers]

    # Add the equities
    add_equities_from_list(tickers=tickers,
                           exchange_codes=exchange_code)

    # Get the equities we just created
    where_str = " ticker in {0}".format(tuple(tickers))
    equities_table_data = db.get_data(table_name='equities',
                                       where_str=where_str)

    # Find the index mapping
    indices = db.get_data(table_name='equity_indices')
    index_id = indices[indices['ticker'] == ticker]['index_id'].values[0]

    # Get index members table
    index_members_table = db.get_table(table_name='equity_index_members')
    index_membership_data = pd.DataFrame(
        columns=index_members_table.columns.keys())
    index_membership_data['equity_id'] = equities_table_data['id']
    index_membership_data['valid_date'] = dt.date.today()
    index_membership_data['index_id'] = index_id

    # Update equity index membership table
    db.execute_db_save(df=index_membership_data, table=index_members_table)


def add_equities_to_security_universe(tickers=None,
                                      exchange_codes=None,
                                      method='figi'):

    security_type = 'equity'
    securities_table = db.get_table(table_name='securities')

    mapping = None

    if method == 'figi':

        # Figi has a batch size per minute restriction
        batch_size = 100
        delay_seconds = 45

        num_batches = int(np.ceil(float(len(tickers)) / batch_size))

        for i in range(0, num_batches):

            logging.info("mapping to OpenFIGI: batch " + str(i)
                         + " out of " + str(num_batches))

            if len(tickers) == 1:
                batch_tickers = tickers
                batch_exchange_codes = exchange_codes
            else:
                lrange = i * batch_size
                urange = np.min([(i + 1) * batch_size, len(tickers) - 1])
                batch_tickers = tickers[lrange:urange]
                batch_exchange_codes = exchange_codes[lrange:urange]
            mapping = FigiApi.retrieve_mapping(id_type='TICKER',
                                               ids=batch_tickers,
                                               exch_codes=batch_exchange_codes)

            mapping['security_type'] = security_type

            cols = ['figi_id', 'composite_figi_id', 'bbg_sector', 'exchange_code',
                    'security_type', 'security_sub_type', 'ticker', 'name']

            mapping = mapping[cols]

            db.execute_db_save(df=mapping,
                               table=securities_table,
                               extra_careful=False)

            time.sleep(delay_seconds)


def add_equities_from_list(tickers=None,
                           exchange_codes=None):

    # Add to securities universe
    add_equities_to_security_universe(tickers=tickers,
                                      exchange_codes=exchange_codes)

    df = pd.DataFrame(data=tickers, columns=['ticker'])
    df['exchange_code'] = exchange_codes
    equities_table = db.get_table(table_name='equities')

    # Add to equities table
    db.execute_db_save(df=df,
                       table=equities_table,
                       extra_careful=False,
                       use_column_as_key='ticker')


"""
-------------------------------------------------------------------------------
EQUITY INDICES
-------------------------------------------------------------------------------
"""


def ingest_equity_index_mappings():

    yahoo_mappings = YahooApi.get_equity_index_identifiers()
    quandl_mappings = QuandlApi.get_equity_index_identifiers()

    # Get tables
    index_identifiers_table = db.get_table('index_identifiers')

    # Get data source ID's
    data_sources = db.read_sql("select * from data_sources")
    data_sources.index = data_sources['id']
    yahoo_id = data_sources[data_sources['name'] == 'yahoo']['id'].values[0]
    quandl_id = data_sources[data_sources['name'] == 'quandl']['id'].values[0]

    # Get index ID's
    index_ids = db.get_data("equity_indices")
    index_ids.index = index_ids['ticker']

    df = pd.DataFrame(columns=index_identifiers_table.columns.keys())

    counter = 0
    for ticker in yahoo_mappings.keys():
        df.loc[counter, 'data_source_id'] = yahoo_id
        df.loc[counter, 'index_id'] = index_ids.loc[ticker, 'index_id']
        df.loc[counter, 'identifier_type'] = 'ticker'
        df.loc[counter, 'identifier_value'] = yahoo_mappings[ticker]
        counter += 1

    for ticker in quandl_mappings.keys():
        df.loc[counter, 'data_source_id'] = quandl_id
        df.loc[counter, 'index_id'] = index_ids.loc[ticker, 'index_id']
        df.loc[counter, 'identifier_type'] = 'ticker'
        df.loc[counter, 'identifier_value'] = quandl_mappings[ticker]
        counter += 1

    db.execute_db_save(df=df,
                       table=index_identifiers_table,
                       extra_careful=False)


def ingest_historical_index_prices(ids=None,
                                   start_date=default_start_date,
                                   end_date=dt.datetime.today(),
                                   data_source_name='yahoo',
                                   batch=True,
                                   batch_size=20):

    # Get all indices
    indices = db.get_equity_indices()

    # Default is everything
    if ids is None:
        ids = indices['index_id'].tolist()

    # Map to data source ID's
    identifiers = db.get_index_data_source_identifiers(
        data_source_name=data_source_name,
        tickers=indices['ticker'].tolist())

    # Filter down to requested set
    identifiers = identifiers[identifiers['index_id'].isin(ids)]

    # Data source ID's
    data_source_ids = identifiers['identifier_value'].values.tolist()
    data_source_ids = [str(id_) for id_ in data_source_ids]

    if data_source_name == 'yahoo':
        prices = YahooApi.retrieve_prices(
            equity_tickers=data_source_ids,
            start_date=start_date,
            end_date=end_date)
    elif data_source_name == 'quandl':
        prices = QuandlApi.get_data(
            tickers=data_source_ids,
            start_date=start_date,
            end_date=end_date)

    prices.head()

    index_prices_table = db.get_table('equity_index_prices')
    prices = prices.reset_index()
    prices = prices.rename(columns={'Date': 'date',
                                    'minor': 'identifier_value'})

    prices_df = pd.merge(left=identifiers[['index_id', 'identifier_value']],
                         right=prices,
                         on='identifier_value')
    prices_df = prices_df.rename(columns={'index_id': 'id'})

    for col in prices_df:
        if col not in index_prices_table.columns.keys():
            del prices_df[col]

    prices_df = prices_df[index_prices_table.columns.keys()]

    db.execute_db_save(df=prices_df,
                       table=index_prices_table,
                       extra_careful=False,
                       time_series=True)


"""
-------------------------------------------------------------------------------
EQUITY IMPLIED VOLATILITIES
-------------------------------------------------------------------------------
"""


def process_historical_orats_data(tickers=None):

    if tickers is None:

        d = db.read_sql("select distinct id, ticker from staging_orats")
        tickers = [str(t) for t in d['ticker'].values.tolist()]
        tickers.index = tickers['id']

    for ticker in tickers:

        data = md.get_equity_implied_volatility(tickers=[ticker],
                                                start_date=default_start_date)

        data['equity_id'] = tickers.loc[ticker, 'id']

        x=1


def ingest_historical_orats_data(full_history=False):

    data = QuandlApi.get_orats_data(full_history=full_history)
    table = db.get_table("staging_orats")

    if not full_history:
        max_date = data['date'].max()
        offset = 3
        cutoff_date = max_date - BDay() * offset
        data = data[data['date'] >= cutoff_date]

    db.execute_db_save(df=data,
                       table=table,
                       time_series=True,
                       extra_careful=False)


def ingest_historical_orats_data_from_api(start_date=default_start_date):

    """
    This uses the slow API calls to request historical ORATS data from Quandl.
    Its use is necessary when partial historical data that isn't just the
    most recent data is desired.
    :param start_date: DateTime
    :return: none
    """

    days_offset = 21
    cutoff_date = start_date - days_offset * BDay()

    table = db.get_table("staging_orats")
    s = "select distinct ticker from staging_orats " \
        " where date >= '{0}'".format(str(cutoff_date))
    s += ' order by ticker asc'
    tmp = db.read_sql(s)
    tickers = tmp['ticker'].values.tolist()
    tickers = [str(ticker) for ticker in tickers]

    batch_size = 10
    num_batches = int(np.ceil(float(len(tickers)) / batch_size))

    for i in range(0, num_batches):

        logging.info("batch " + str(i) + " out of " + str(num_batches))

        lrange = i * batch_size
        urange = np.min([(i + 1) * batch_size, len(tickers) - 1])

        if batch_size == 1:
            batch_ids = [tickers[lrange]]
        else:
            batch_ids = tickers[lrange:urange]

        data = QuandlApi.get_orats_data_from_api(batch_ids,
                                                 start_date,
                                                 end_date=dt.datetime.today())

        if data is None:
            continue

        data = data[np.isfinite(data['iv_1m'])]

        if len(data) > 0:
            db.execute_db_save(df=data,
                               table=table,
                               time_series=True,
                               extra_careful=False)


"""
-------------------------------------------------------------------------------
FUTURES OPTIONS IMPLIED VOLATILITIES
-------------------------------------------------------------------------------
"""


def process_optionworks_mappings():

    """
    This must be run after process_optionworks_series. It maps OW series
    to internal futures series.
    :return: None
    """

    s = "select distinct ow_series_id, exchange_code, futures_series" \
        " from optionworks_codes"
    d = db.read_sql(s)

    futures_series = db.get_data(table_name='futures_series')
    futures_series = futures_series[['id', 'series', 'exchange']]

    tmp = pd.merge(left=d,
                   right=futures_series,
                   how='left',
                   right_on=['series', 'exchange'],
                   left_on=['futures_series', 'exchange_code'])

    # Internal (quandl) series to optionworks series
    # Annoying because quandl codes just say "CME" where OW more detailed
    manual_mapping = {'CME_LH': 1133, # Lean hogs
                      'CBT_SM': 1220, # Soybean meal
                      'CMX_SI': 1219, # Silver
                      'CBT_C': 1007,  # Corn
                      'NYX_RC': 1121, # Robusta coffee
                      'NYM_RB': 1196, # Rbob
                      'CBT_TY': 1234, # T-note
                      'NYM_HO': 1096, # Heating oil
                      'CBT_BO': 1002, # Soybean oil
                      'CBT_W': 1247,  # Wheat
                      'CBT_FF': 1059, # Fed funds
                      'CBT_FV': 1076, # 5-year T-note
                      'CBT_S': 1212,  # Soybeans
                      'NYX_W': 1213,  # Sugar
                      'NYM_NG': 1154, # Natural gas
                      'CMX_HG': 1093, # Copper
                      'CBT_US': 1242, # US long bond
                      'NYM_CL': 1015, # NYMEX crude
                      'CMX_GC': 1080, # Gold
                      'NYX_C': 1011,  # Cocoa
                      'CBT_TU': 1233, # 2-year T-note
                      }
    tmp.index = tmp['ow_series_id']
    for series in manual_mapping:
        tmp.loc[series, 'id'] = manual_mapping[series]

    mapping_data = tmp.reset_index(drop=True)
    mapping_data = mapping_data[['ow_series_id', 'id']]
    mapping_data['source'] = 'optionworks'
    mapping_data['id'] = mapping_data['id'].replace(np.nan,
                                                    constants.invalid_value)
    mapping_data = mapping_data[mapping_data['id'] != constants.invalid_value]
    mapping_data = mapping_data.rename(columns={'ow_series_id': 'source_id',
                                                'id': 'series_id'})

    table = db.get_table('futures_series_identifiers')
    db.execute_db_save(df=mapping_data, table=table, extra_careful=False)

    # NYX_T is NYX feed wheat
    # NYX_ECO is rapeseed
    # NYX_EBM is milling wheat
    # NYX_EMA is corn EMA
    # NYX_EOB is malting barley
    # ICE_RS is canola


def process_optionworks_series():

    """
    This grabs data from the OptionWorks staging tables and processes the data
    series there, creating intelligent metadata for OptionWorks
    :return: none
    """

    ow_constant_maturities = {
        '1W': 5,
        '1M': 21,
        '2M': 42,
        '3M': 63,
        '6M': 126,
        '9M': 189,
        '1Y': 252,
        '2Y': 512,
        '5Y': 252 * 5
    }

    for ow_data_type in ['ivm', 'ivs']:

        s = 'select distinct code from staging_optionworks_' \
            + ow_data_type + ' order by code asc'
        d = db.read_sql(s)

        d_ = d['code'].str.split('_')
        ow_cols = ['exchange_code', 'futures_series',
                   'option', 'maturity_str', 'ow_data_type']
        columns = ow_cols + ['ow_code', 'maturity_type',
                             'futures_contract', 'days_to_maturity']
        df = pd.DataFrame(index=d_.index, columns=columns)
        df['ow_code'] = d['code']
        i = 0
        for row in d_:
            df.loc[i, ow_cols] = row
            mat_str = df.loc[i, 'maturity_str']
            if len(mat_str) == 2:
                df.loc[i, 'days_to_maturity'] = \
                    str(ow_constant_maturities[mat_str])
                df.loc[i, 'maturity_type'] = 'constant_maturity'
            elif len(mat_str) == 5:
                df.loc[i, 'futures_contract'] = df.loc[
                    i, 'futures_series'] + mat_str
                df.loc[i, 'maturity_type'] = 'futures_contract'
            i += 1
        df = df[columns].where((pd.notnull(df)), None)

        table = db.get_table('optionworks_codes')
        db.execute_db_save(df=df, table=table, extra_careful=False)


def impute_futures_maturity_dates_from_optionworks(futures_contract_tickers=None):

    # Update futures maturity dates from optionworks
    ow_series = db.get_data(table_name='futures_series_identifiers',
                            where_str="source = 'optionworks'")
    ows = tuple(ow_series['series_id'].values)

    where_str = "  series_id in {0}".format(ows)
    if futures_contract_tickers is not None:
        where_str += " and ticker in {0}".format(
            dbutils.format_for_query(futures_contract_tickers))

    fc = db.get_data(table_name='futures_contracts', where_str=where_str)

    s = " select distinct on(futures_contract)" \
        " date, futures_contract, dtt " \
        " from futures_ivol_fixed_maturity_surface_model "
    d = db.read_sql(s)

    df = pd.merge(left=fc,
                  right=d,
                  left_on='ticker',
                  right_on='futures_contract')
    df['imputed_maturity_date'] = pd.to_datetime(df['date']) \
                                  + pd.TimedeltaIndex(df['dtt'], unit='D')
    df['maturity_date'] = df['imputed_maturity_date']
    df = df[['id', 'series_id', 'ticker', 'maturity_date']]

    table = db.get_table('futures_contracts')
    db.execute_db_save(df=df, table=table)


def process_all_optionworks_data(start_date=default_start_date):

    # This gets the master list of optionworks series that we have mapped
    d = db.read_sql("select * from futures_series_identifiers"
                    " where source = 'optionworks' ")
    ow_series = d['source_id'].values.tolist()
    series_ids = d['series_id'].values.tolist()
    ow_series = [str(s) for s in ow_series]

    # s = 'select distinct futures_series from optionworks_codes ' \
    #     'where ow_series_id in {0}'.format(
    #         dbutils.format_for_query(ow_series))
    # d = db.read_sql(s)
    # d = [str(d_) for d_ in d['futures_series'].values]

    counter = 0
    for series_id in series_ids:

        logging.info(" processing " + ow_series[counter] + " ... ")

        process_optionworks_data_ivm(series_id=series_id,
                                     start_date=start_date,
                                     maturity_type='constant_maturity')

        process_optionworks_data_ivm(series_id=series_id,
                                     start_date=start_date,
                                     maturity_type='futures_contract')

        process_optionworks_data_ivs(series_id=series_id,
                                     start_date=start_date,
                                     maturity_type='constant_maturity')

        process_optionworks_data_ivs(series_id=series_id,
                                     start_date=start_date,
                                     maturity_type='futures_contract')
        counter += 1


def preprocess_optionworks_data(series_id=None,
                                ow_data_type=None,
                                maturity_type=None,
                                source_table_name=None,
                                target_table_name=None,
                                start_date=None):

    # TODO: what to do with weekly options?

    if maturity_type not in ('futures_contract', 'constant_maturity'):
        raise ValueError('maturity_type must be futures_contract or '
                         'constant_maturity.')

    if ow_data_type not in ('IVM', 'IVS'):
        raise ValueError('ow_data_type must be IVM or IVS.')

    if start_date is None:
        start_date = dt.datetime.today() - BDay()
    target_table = db.get_table(target_table_name)

    # Get the ow_series_id for the series_id
    ow_series_id = db.get_data(
        table_name='futures_series_identifiers',
        where_str='series_id = {0}'.format(series_id) +
                  " and source = 'optionworks'")['source_id'].values[0]

    # Get the basic metadata
    s = 'select distinct ow_code, futures_series, exchange_code, ow_series_id, '
    if maturity_type == 'futures_contract':
        s += ' futures_contract '
    elif maturity_type == 'constant_maturity':
        s += ' days_to_maturity '
    s += ' from optionworks_codes '
    s += " where ow_series_id = '{0}'".format(ow_series_id)
    s += " and active = True "
    s += " and option = futures_series "
    metadata = db.read_sql(s)
    metadata['series_id'] = series_id

    # Get the optionworks codes we'll need to request vol data
    # from the staging tables
    e = db.read_sql(
        "select ow_code, days_to_maturity from optionworks_codes " +
        " where ow_series_id = '{0}'".format(ow_series_id) +
        " and ow_data_type = '{0}'".format(ow_data_type) +
        " and maturity_type = '{0}'".format(maturity_type) +
        " and active = True "
        " and option = futures_series "
        " order by days_to_maturity asc ")
    codes = [str(code) for code in e['ow_code'].values.tolist()]

    # If we don't find anything, oops return
    if len(codes) == 0:
        logging.info("no matching codes found!")
        return None, None

    # Get the actual implied volatility data
    s = 'code in {0}'.format(dbutils.format_for_query(codes)) + \
        " and date >= '{0}'".format(start_date)
    data = db.get_data(table_name=source_table_name,
                       where_str=s)

    df = pd.merge(left=metadata,
                  right=data,
                  left_on='ow_code',
                  right_on='code')

    return df, target_table


def process_optionworks_data_ivs(series_id=None,
                                 start_date=None,
                                 maturity_type=None):

    if maturity_type not in ('constant_maturity', 'futures_contract'):
        raise ValueError('maturity_type must be '
                         'constant_maturity or futures_contract')

    ow_data_type = 'IVS'
    source_table_name = 'staging_optionworks_' + ow_data_type.lower()

    target_table_name = 'futures_ivol_'
    if maturity_type == 'futures_contract':
        target_table_name += 'fixed_maturity_by_delta'
        maturity_col = 'futures_contract'
    elif maturity_type == 'constant_maturity':
        target_table_name += 'constant_maturity_by_delta'
        maturity_col = 'days_to_maturity'

    df, target_table = preprocess_optionworks_data(
        series_id=series_id,
        ow_data_type=ow_data_type,
        maturity_type=maturity_type,
        source_table_name=source_table_name,
        target_table_name=target_table_name,
        start_date=start_date)

    if df is None:
        w = "warning: did not find OW " + str(series_id) + " data to process!"
        logging.warn(w)
        return

    ow_ivs_cols = QuandlApi.get_optionworks_ivs_cols()
    call_cols = [s.lower() for s in ow_ivs_cols
                 if s[0] == 'C' and s != 'Code']
    put_cols = [s.lower() for s in ow_ivs_cols if s[0] == 'P']

    call_data = df[['series_id', maturity_col, 'date'] + call_cols]
    put_data = df[['series_id', maturity_col, 'date'] + put_cols]

    call_data['option_type'] = 'call'
    put_data['option_type'] = 'put'

    index_cols = ['series_id', maturity_col, 'option_type', 'date']
    call_data = call_data.set_index(keys=index_cols, drop=True)
    put_data = put_data.set_index(keys=index_cols, drop=True)

    # rename columns as delta
    for col in call_cols:
        call_data = call_data.rename(columns={col: 'ivol_' + col[1:3] + 'd'})
    for col in put_cols:
        put_data = put_data.rename(columns={col: 'ivol_' + col[1:3] + 'd'})

    data = put_data.append(call_data)

    db.execute_db_save(df=data,
                       table=target_table,
                       extra_careful=False,
                       time_series=True)


def process_optionworks_data_ivm(series_id=None,
                                 start_date=None,
                                 maturity_type=None):

    ow_data_type = 'IVM'
    source_table_name = 'staging_optionworks_' + ow_data_type.lower()
    target_table_name = 'futures_ivol_'
    if maturity_type == 'constant_maturity':
        target_table_name += 'constant_maturity_surface_model'
    elif maturity_type == 'futures_contract':
        target_table_name += 'fixed_maturity_surface_model'

    df, target_table = preprocess_optionworks_data(
        series_id=series_id,
        ow_data_type=ow_data_type,
        maturity_type=maturity_type,
        source_table_name=source_table_name,
        target_table_name=target_table_name,
        start_date=start_date)

    # Failed to find data
    if df is None:
        return

    df = df[target_table.columns.keys()]
    df['date'] = pd.to_datetime(df['date'])

    db.execute_db_save(df=df,
                       table=target_table,
                       extra_careful=False,
                       time_series=True)


def ingest_optionworks_data(full_history=False, start_date=None):

    """

    :rtype: object
    """
    data_ivm, data_ivs = QuandlApi.get_optionworks_data(full_history)

    if start_date is not None:
        data_ivm = data_ivm[data_ivm['date'] >= start_date]
        data_ivs = data_ivs[data_ivs['date'] >= start_date]

    table_name_ivm = "staging_optionworks_ivm"
    table_name_ivs = "staging_optionworks_ivs"

    s = "select max(date) from " + table_name_ivm
    d = db.read_sql(s)
    latest_date = pd.to_datetime(d['max'].values[0])
    retrieve_date = data_ivm['date'].max()

    if start_date is None and retrieve_date <= latest_date:
        logging.info("Retrieval date is on or before existing data!")
        return

    # This is sort of hokey... they keep returning old stuff
    if not full_history:
        data_ivm = data_ivm[data_ivm['date'] == retrieve_date]
        data_ivs = data_ivs[data_ivs['date'] == data_ivs['date'].max()]

    table = db.get_table(table_name_ivm)
    db.execute_db_save(df=data_ivm, table=table, time_series=True)

    table = db.get_table(table_name_ivs)
    db.execute_db_save(df=data_ivs, table=table, time_series=True)

    # database_table_path = db.connection_string + "::" + table_name_ivm
    # odo.odo(data_ivm, database_table_path)
    #
    # database_table_path = db.connection_string + "::" + table_name_ivs
    # odo.odo(data_ivs, database_table_path)


"""
-------------------------------------------------------------------------------
DIVIDENDS
-------------------------------------------------------------------------------
"""


def ingest_historical_dividends(ids=None,
                                start_date=None,
                                end_date=None,
                                data_source='yahoo'):

    if end_date is None:
        end_date = dt.datetime.today()

    if start_date is None:
        start_date = default_start_date

    # Use existing routine to get tickers and id defaults
    equities, equity_prices_table, ids, tickers, rows = \
        _prep_equity_price_ingest(ids)

    if data_source == 'yahoo':

        dividends = YahooApi.retrieve_dividends(equity_tickers=tickers,
                                                start_date=start_date,
                                                end_date=end_date)

        schedules_table = db.get_table(table_name='equity_schedules')
        equities = db.get_data(table_name='equities')

        schedules_data = pd.DataFrame(columns=schedules_table.columns.keys())
        schedules_data['ticker'] = dividends['Ticker']
        schedules_data['date'] = dividends['Date']
        schedules_data['value'] = dividends['Dividend']
        schedules_data['schedule_type'] = 'dividend'
        del schedules_data['id']

        schedules_data = pd.merge(left=schedules_data,
                                  right=equities,
                                  on='ticker')

    else:
        raise NotImplementedError

    db.execute_db_save(df=schedules_data, table=schedules_table)


"""
-------------------------------------------------------------------------------
CFTC Positioning
-------------------------------------------------------------------------------
"""


def map_cftc_codes():
    mapping = pd.read_excel("data/cftc_qfl_code_mapping.xlsx")
    mapping = mapping.where((pd.notnull(mapping)), None)
    mapping = mapping.rename(columns={'map_series': 'series'})

    futures_series = db.get_data('futures_series')
    del futures_series['cftc_code']

    futures_series_mapped = pd.merge(
        left=futures_series,
        right=mapping[['series', 'CFTC_Contract_Market_Code']],
        on='series')
    futures_series_mapped = futures_series_mapped.rename(
        columns={'CFTC_Contract_Market_Code': 'cftc_code'})

    table = db.get_table('futures_series')
    db.execute_db_save(df=futures_series_mapped,
                       table=table,
                       extra_careful=False)


def ingest_cftc_positioning_data(category=None, full_history=False):
    """
    Ingest CFTC positioning data
    :param category: string "commodity" or "financial"
    :param full_history: bool
    :return: None
    """

    if not category in ["commodity", "financial"]:
        raise ValueError("category must be 'commodity' or 'financial'")

    filename = "data/cftc_positioning.csv"
    table_name = "staging_cftc_positioning_" + category

    if category == "commodity":
        data = DataScraper.retrieve_cftc_commodity_positioning_data(
            full_history=full_history)
    elif category == "financial":
        data = DataScraper.retrieve_cftc_financial_positioning_data(
            full_history=full_history)

    for col in data.columns:
        data = data.rename(columns={col: col.lower()})

    # Weekly update
    if not full_history:
        s = 'select max(report_date_as_mm_dd_yyyy) ' \
            'from staging_cftc_positioning_' + category
        d = db.read_sql(s)['max'].values[0]
        data = data[data['report_date_as_mm_dd_yyyy'] > d]
    data.to_csv(filename)

    database_table_path = db.connection_string + "::" + table_name
    odo.odo(data, database_table_path)

    os.remove(filename)


"""
-------------------------------------------------------------------------------
VOLATILITY INDICES
-------------------------------------------------------------------------------
"""


def ingest_historical_volatility_index_prices(start_date=default_start_date):

    # Take start-date back a few days to ensure all data found
    start_date = start_date - 5 * BDay()

    table = db.get_table('generic_index_prices')

    # VIX
    data = YahooApi.retrieve_prices("^VIX", start_date)
    data = data.reset_index()
    data = data.rename(columns={'Date': 'date'})
    s = "select id from generic_indices where ticker = 'VIX'"
    id = db.read_sql(s).iloc[0].values[0]
    data['id'] = id
    if len(data) > 0:
        db.execute_db_save(df=data,
                           table=table,
                           extra_careful=False,
                           time_series=True)

    # V2X
    data = DataScraper.retrieve_vstoxx_historical_prices()
    s = "select id from generic_indices where ticker = 'V2X'"
    id = db.read_sql(s).iloc[0].values[0]
    data['id'] = id
    del data['ticker']
    if len(data) > 0:
        db.execute_db_save(df=data,
                           table=table,
                           extra_careful=False,
                           time_series=True)

    # SKEW
    data = QuandlApi.get('CBOE/SKEW', start_date).reset_index()
    s = "select id from generic_indices where ticker = 'SKEW'"
    id = db.read_sql(s).iloc[0].values[0]
    data['id'] = id
    data = data.rename(columns={'SKEW': 'last_price', 'Date': 'date'})
    if len(data) > 0:
        db.execute_db_save(df=data,
                           table=table,
                           extra_careful=False,
                           time_series=True)


def add_volatility_futures_series():

    # Manual because doesn't exist in Quandl source
    futures_series_table = db.get_table('futures_series')

    # VIX
    df = pd.DataFrame(columns=futures_series_table.columns.keys(), index=[0])
    df['series'] = 'VX'
    df['description'] = 'CBOE VIX Index'
    df['exchange'] = 'CBOE'
    df['currency'] = 'USD'
    df['contract_size'] = 'USD1000 x Index'
    df['units'] = 'Points'
    df['point_value'] = 1000
    df['tick_value'] = 50
    df['delivery_months'] = 'FGHJKMNQUVXZ'
    del df['id']
    db.execute_db_save(df=df, table=futures_series_table)

    # V2X
    df = pd.DataFrame(columns=futures_series_table.columns.keys(), index=[0])
    df['series'] = 'FVS'
    df['description'] = 'EUREX VSTOXX Index'
    df['exchange'] = 'EUREX'
    df['currency'] = 'EUR'
    df['contract_size'] = 'EUR100 x Index'
    df['units'] = 'Points'
    df['point_value'] = 100
    df['tick_value'] = 5
    df['delivery_months'] = 'FGHJKMNQUVXZ'
    del df['id']
    db.execute_db_save(df=df, table=futures_series_table)


def update_vix_futures_settle_prices():

    offset_hours = 1.0
    series = "VX"

    if _is_after_close(offset_hours=offset_hours):
        overwrite_date = dt.datetime.today().date()
    else:
        overwrite_date = utils.workday(dt.datetime.today(), -1).date()

    overwrite_date = pd.to_datetime(overwrite_date)
    prices = DataScraper.update_vix_settle_price(overwrite_date)

    futures_contracts = md.get_futures_contracts_by_series(
        futures_series=series, start_date=overwrite_date)

    prices['maturity_date'] = pd.to_datetime(prices['maturity_date'])

    data = pd.merge(left=futures_contracts, right=prices, on='maturity_date')
    data = data[['id', 'date', 'settle_price']]

    table = db.get_table('futures_prices')
    db.execute_db_save(df=data, table=table, extra_careful=False)

    # Map to the generic tickers
    tickers = ["VX" + str(i + 1) for i in range(0, len(data))]
    data['ticker'] = tickers

    # Get contract ID's
    s = "select * from generic_futures_contracts where series_id = " \
        " ( select id from futures_series where series = 'VX')"
    s += " order by contract_number asc"
    generic_futures_contracts = db.read_sql(s)

    generic_data = pd.merge(left=generic_futures_contracts[['id', 'ticker']],
                            right=data[['ticker', 'date', 'settle_price']],
                            on='ticker')

    del generic_data['ticker']

    table = db.get_table('generic_futures_prices')
    db.execute_db_save(df=generic_data, table=table, extra_careful=False)


"""
-------------------------------------------------------------------------------
FUTURES
-------------------------------------------------------------------------------
"""


def add_futures_series():

    futures_series = qfl_data.QuandlApi.get_futures_universe()

    # Drop inactive stuff
    futures_series = futures_series[futures_series['Active'] == 1]

    # Rename
    # Note the typo in their column "Delivery Months" - they may fix this
    rename_dict = {'Symbol': 'series',
                   'Exchange': 'exchange',
                   'Name': 'description',
                   'Full Point Value': 'point_value',
                   'Currency': 'currency',
                   'Contract Size': 'contract_size',
                   'Units': 'units',
                   'Tick Value': 'tick_value',
                   'Trading Times': 'trading_times',
                   'Delievery Months': 'delivery_months',
                   'Delivery Months': 'delivery months',
                   'Start Date': 'start_date'
                   }
    futures_series = futures_series.rename(columns=rename_dict)

    # Filter down columns
    futures_series_table = db.get_table('futures_series')
    for col in futures_series.columns:
        if col not in futures_series_table.columns.keys():
            del futures_series[col]

    # Unique (contracts, not session times)
    futures_series = futures_series.drop_duplicates(
        subset=['series', 'exchange'])

    db.execute_db_save(df=futures_series, table=futures_series_table,
                       extra_careful=False, time_series=False,
                       use_column_as_key='series')


def ingest_historical_generic_futures_prices(futures_series=None,
                                             exchange_code=None,
                                             source_series=None,
                                             contract_range=None,
                                             dataset=None,
                                             start_date=None):

    futures_data = QuandlApi.retrieve_historical_generic_futures_prices(
        start_date=start_date,
        futures_series=futures_series,
        source_series=source_series,
        contract_range=contract_range,
        dataset=dataset)

    if len(futures_data) == 0:
        return

    futures_data.index.names = ['ticker', 'date']
    futures_data = futures_data.reset_index()

    # Contracts
    futures_contracts_table = db.get_table(
        table_name='generic_futures_contracts')
    contracts_retrieved = pd.DataFrame(futures_data.groupby('ticker')
                                       .last()['contract_number'])

    futures_series_data = db.get_futures_series(futures_series=futures_series,
                                                exchange_code=exchange_code)

    series_id = futures_series_data['id'].values[0]
    contracts_retrieved['series_id'] = series_id
    contracts_retrieved = contracts_retrieved.reset_index()

    db.execute_db_save(df=contracts_retrieved,
                       table=futures_contracts_table,
                       use_column_as_key='ticker',
                       extra_careful=False)

    # Prices
    futures_prices_table = db.get_table(table_name='generic_futures_prices')

    # Filter for valid prices
    futures_data = futures_data[
        futures_data['settle_price'] > 0]

    # Join with id
    futures_contracts = db.get_data(table_name='generic_futures_contracts')

    final_data = pd.merge(left=futures_contracts[['id', 'ticker']],
                          right=futures_data,
                          on='ticker')

    for col in final_data.columns:
        if col not in futures_prices_table.columns.keys():
            del final_data[col]

    final_data = utils.replace_nan_with_none(final_data)

    result = db.execute_db_save(df=final_data,
                                table=futures_prices_table,
                                extra_careful=False,
                                time_series=True)
    return result


def precompute_seasonality_adjusted_vol_futures_prices(futures_series=None,
                                                       start_date=None,
                                                       price_field='settle_price'):
    vix_spot_tenor = 30
    vix_base_trading_days = 23
    trading_days_adj_factor = 0.5
    december_effect_vol_points = 0.25

    generic_futures_data = md.get_generic_futures_prices_by_series(
        futures_series=futures_series,
        start_date=start_date)

    # VIX-specific logic for adjusting calendar day-count conventions
    futures_contracts = md.get_futures_contracts_by_series(futures_series)

    generic_futures_prices = calcs.compute_seasonality_adj_vol_futures_prices(
        futures_data=generic_futures_data,
        futures_contracts=futures_contracts,
        vix_spot_tenor=vix_spot_tenor,
        base_trading_days=vix_base_trading_days,
        december_effect_vol_points=december_effect_vol_points,
        trading_days_adj_factor=trading_days_adj_factor,
        price_field=price_field
    )

    futures_prices_to_update = generic_futures_prices[
        ['id', 'seasonality_adj_price']].reset_index()
    del futures_prices_to_update['ticker']

    if len(futures_prices_to_update) > 0:
        table = db.get_table('generic_futures_prices')
        db.execute_db_save(df=futures_prices_to_update,
                           table=table,
                           extra_careful=False,
                           time_series=True)

    # Now do standard futures time series data
    futures_prices_by_contract = md.get_futures_prices_by_series(
        futures_series=futures_series, start_date=start_date) \
        .reset_index()

    # Delete column, we're going to replace
    del futures_prices_by_contract['seasonality_adj_price']

    cols = ['date', 'contract_ticker', 'seasonality_adj_price']
    generic_futures_prices[
        'date'] = generic_futures_prices.index.get_level_values('date')
    futures_prices_by_contract = pd.merge(
        left=futures_prices_by_contract,
        right=generic_futures_prices[cols],
        left_on=['date', 'ticker'],
        right_on=['date', 'contract_ticker']
    )
    futures_prices_by_contract = futures_prices_by_contract[
        ['id', 'date', 'seasonality_adj_price']]

    if len(futures_prices_by_contract) > 0:
        table = db.get_table('futures_prices')
        db.execute_db_save(df=futures_prices_by_contract,
                           table=table,
                           extra_careful=False,
                           time_series=True)


def precompute_constant_maturity_futures_prices(futures_series=None,
                                                start_date=None,
                                                constant_maturities_in_days=None,
                                                price_field='settle_price',
                                                spot_prices=None,
                                                volatilities=False):

    generic_futures_data = md.get_generic_futures_prices_by_series(
        futures_series=futures_series,
        start_date=start_date)

    if futures_series == 'FVS':
        x=1

    cmfp = calcs.compute_constant_maturity_futures_prices(
        generic_futures_data=generic_futures_data,
        constant_maturities_in_days=constant_maturities_in_days,
        price_field=price_field,
        spot_prices=spot_prices,
        volatilities=volatilities
    )

    cmfp = cmfp.stack() \
        .reset_index() \
        .rename(columns={'level_1': 'days_to_maturity', 0: 'price'})

    cmfp['series_id'] = db.get_futures_series(futures_series).iloc[0]['id']

    if len(cmfp) == 0:
        logging.info("No data to archive!")
        return

    table = db.get_table('constant_maturity_futures_prices')

    db.execute_db_save(df=cmfp,
                       table=table,
                       extra_careful=False,
                       time_series=True)


def precompute_historical_futures_days_to_maturity(futures_series=None,
                                                   start_date=None,
                                                   generic=False):

    # These are neede for the map either way
    futures_prices = md.get_generic_futures_prices_by_series(
        futures_series=futures_series,
        start_date=start_date,
        mapped_view=False)

    # Get contract map (it's going to be empty for new stuff)
    contract_map, futures_contracts = md.get_futures_generic_contract_map(
        futures_series=futures_series,
        futures_prices=futures_prices,
        price_field='settle_price',
        start_date=start_date)

    if generic:
        table_name = 'generic_futures_prices'
    else:
        table_name = 'futures_prices'
        futures_prices = md.get_futures_prices_by_series(
            futures_series=futures_series,
            start_date=start_date)

    if len(futures_prices) == 0:
        return

    # Overwrite contract ticker and maturity date
    futures_contracts = futures_contracts.rename(columns={'ticker':
                                                          'contract_ticker'})

    contract_map = pd.DataFrame(contract_map.stack(level='ticker'),
                                columns=['contract_ticker'])

    contract_map = pd.merge(left=contract_map.reset_index(),
                            right=futures_contracts,
                            on='contract_ticker')

    if len(contract_map) == 0:
        logging.info("Could not find any matching contracts!")
        return

    cols = ['contract_ticker', 'maturity_date', 'futures_contract_id']
    if not generic:
        del contract_map['ticker']
        contract_map = contract_map.rename(columns={'contract_ticker': 'ticker'})
        cols = ['maturity_date', 'futures_contract_id']
    contract_map = contract_map.set_index(['date', 'ticker'])
    futures_prices[cols] = contract_map[cols]

    # Retrieve the series and the holiday calendar
    calendar_name = md.get_futures_calendar_name(futures_series)

    # Calculate net workdays
    start_dates = futures_prices.index.get_level_values('date')
    end_dates = futures_prices['maturity_date'].values
    futures_prices['days_to_maturity'] = utils.networkdays(
            start_date=start_dates,
            end_date=end_dates,
            calendar_name=calendar_name)


    # Keep only futures prices columns
    table = db.get_table(table_name)
    futures_prices = futures_prices.reset_index()
    for col in futures_prices.columns:
        if col not in table.columns.keys():
            del futures_prices[col]

    db.execute_db_save(df=futures_prices,
                       table=table,
                       time_series=True,
                       extra_careful=False)


def ingest_historical_futures_universe():

    df = QuandlApi.get_futures_metadata()
    contract_range = range(1, 10)

    for i in range(0, len(df)):

        futures_series = df.loc[i, 'futures_series']
        contracts_dataset = df.loc[i, 'contracts_dataset']
        generic_dataset = df.loc[i, 'generic_dataset']
        contracts_series = df.loc[i, 'contracts_series']
        generic_series = df.loc[i, 'generic_series']
        start_date = df.loc[i, 'start_date']

        logging.info('ingesting historical futures prices...')

        ingest_historical_futures_prices(dataset=contracts_dataset,
                                         futures_series=contracts_series)

        logging.info('ingesting historical generic futures prices...')

        ingest_historical_generic_futures_prices(dataset=generic_dataset,
                                                 futures_series=futures_series,
                                                 source_series=generic_series,
                                                 start_date=start_date,
                                                 contract_range=contract_range)

        logging.info('calculating historical days to maturity...')

        precompute_historical_futures_days_to_maturity(
            futures_series=futures_series,
            start_date=start_date,
            generic=False)

        precompute_historical_futures_days_to_maturity(
            futures_series=futures_series,
            start_date=start_date,
            generic=True)


def ingest_futures_contracts(futures_series=None,
                             futures_data=None):

    """
    This works from futures data
    :param futures_series: string
    :param futures_data: DataFrame with fields 'ticker', 'date'
    :return:
    """

    # Get expiration dates: note this only works for expired contracts
    futures_contracts_retrieved = futures_data.reset_index() \
        .groupby('ticker') \
        .last()['date']
    futures_contracts_retrieved = pd.DataFrame(
        futures_contracts_retrieved).reset_index()
    futures_contracts_retrieved = futures_contracts_retrieved.rename(
        columns={'date': 'maturity_date'})

    # Invalidate maturity date for non-expired contracts
    current_date = dt.datetime.today()
    cutoff_date = current_date - 5 * BDay()
    ind = futures_contracts_retrieved.index[
        futures_contracts_retrieved['maturity_date'] > cutoff_date]
    futures_contracts_retrieved.loc[ind, 'maturity_date']\
        = constants.invalid_date

    # Map to appropriate id
    futures_series_data = db.get_data(table_name='futures_series')
    ind = futures_series_data.index[
        futures_series_data['series'] == futures_series]
    series_id = futures_series_data.loc[ind, 'id'].values[0]
    futures_contracts_retrieved['series_id'] = series_id

    # Futures table
    futures_contracts_table = db.get_table(table_name='futures_contracts')

    # Archive
    db.execute_db_save(df=futures_contracts_retrieved,
                       table=futures_contracts_table,
                       use_column_as_key='ticker')


def ingest_historical_futures_prices(dataset=None,
                                     futures_series=None,
                                     start_date=default_start_date,
                                     end_date=dt.datetime.today(),
                                     add_contracts=False):

    # Need logic to identify data sources etc

    # Get series
    tmp = db.get_futures_series(futures_series)
    contract_months = list(tmp['delivery_months'].values[0])
    contract_months = [str(code) for code in contract_months]

    # Get the big dataset
    futures_data = qfl_data.QuandlApi.retrieve_historical_futures_prices(
        start_date=start_date,
        end_date=end_date,
        futures_series=futures_series,
        dataset=dataset,
        contract_months_list=contract_months)
    futures_data.index.names = ['ticker', 'date']

    # Update our contract table
    if add_contracts:
        ingest_futures_contracts(futures_series=futures_series,
                                 futures_data=futures_data)

    # Prices
    futures_prices_table = db.get_table(table_name='futures_prices')
    futures_data = futures_data.reset_index()
    futures_data = futures_data[futures_data['settle_price'] > 0]

    # Join with id
    futures_contracts = db.get_data(table_name='futures_contracts')

    tmp = pd.merge(left=futures_contracts[['id', 'ticker']],
                   right=futures_data,
                   on='ticker')

    # Filter down columns
    for col in tmp.columns:
        if col not in futures_prices_table.columns.keys():
            del tmp[col]

    result = db.execute_db_save(df=tmp,
                                table=futures_prices_table,
                                extra_careful=False,
                                time_series=True)

    # Days to maturity
    precompute_historical_futures_days_to_maturity(
        futures_series=futures_series,
        start_date=start_date,
        generic=False
    )

    # Days to maturity
    precompute_historical_futures_days_to_maturity(
        futures_series=futures_series,
        start_date=start_date,
        generic=True
    )

    return result


def add_futures_contracts(futures_series=None,
                          tickers=None,
                          maturity_dates=None):

    futures_contracts_table = db.get_table('futures_contracts')
    futures_series_data = db.get_futures_series(futures_series)
    series_id = futures_series_data['id'].values[0]

    df = pd.DataFrame(columns=['series_id', 'ticker', 'maturity_date'],
                      index=tickers)

    df['ticker'] = tickers
    df['maturity_date'] = maturity_dates
    df['series_id'] = series_id

    db.execute_db_save(df=df,
                       table=futures_contracts_table,
                       use_column_as_key='ticker')


def update_futures_prices(start_date=None,
                          end_date=None,
                          data_source='quandl'):

    date = start_date
    if date is None:
        date = dt.datetime.today() - BDay()

    fm = QuandlApi.get_futures_metadata()

    # Do this first
    update_vix_futures_settle_prices()

    for i in range(0, len(fm)):

        futures_series = fm.loc[i, 'futures_series']
        source_series = fm.loc[i, 'contracts_series']
        dataset = fm.loc[i, 'contracts_dataset']
        max_contract = fm.loc[i, 'num_generic_contracts']
        contract_range = np.arange(1, int(max_contract) + 1)

        update_futures_prices_by_series(
            futures_series=futures_series,
            exchange_code=dataset,
            date=date,
            source_series=source_series,
            dataset=dataset,
            contract_range=contract_range)


def update_generic_futures_prices_by_series(futures_series=None,
                                            exchange_code=None,
                                            start_date=None,
                                            data_source_name='quandl',
                                            constant_maturities_in_days=None,
                                            contract_range=np.arange(1, 10)):

    # Defaults
    if constant_maturities_in_days is None:
        constant_maturities_in_days = [0, 5, 10, 21, 42, 63, 21 * 4, 21 * 5,
                                       21 * 6, 21 * 7, 21 * 8]
    if start_date is None:
        start_date = utils.DateUtils.workday(date=dt.datetime.today(),
                                             num_days=-1)

    price_field = 'settle_price'

    # Quandl metadata
    fm = QuandlApi.get_futures_metadata()
    fm.index = fm['futures_series']
    source_series = fm.loc[futures_series, 'generic_series']
    dataset = fm.loc[futures_series, 'generic_dataset']

    # Spot prices for specific series
    spot_prices = None
    if futures_series == 'VX':
        spot_prices = md.get_generic_index_prices("VIX", start_date)\
            ['last_price'].reset_index(level='ticker', drop=True)
    elif futures_series == 'FVS':
        spot_prices = md.get_generic_index_prices('V2X', start_date)\
            ['last_price'].reset_index(level='ticker', drop=True)

    # Prices
    ingest_historical_generic_futures_prices(
        futures_series=futures_series,
        exchange_code=exchange_code,
        source_series=source_series,
        contract_range=contract_range,
        dataset=dataset,
        start_date=start_date)

    # Days to maturity
    precompute_historical_futures_days_to_maturity(
        futures_series=futures_series,
        start_date=start_date,
        generic=True
    )

    # Right now seasonality adjustments only for volatility futures
    volatilities = False
    if futures_series in ('VX', 'FVS'):
        price_field = 'seasonality_adj_price'
        volatilities = True
        precompute_seasonality_adjusted_vol_futures_prices(
            futures_series=futures_series, start_date=start_date
        )

    # Constant-maturity
    precompute_constant_maturity_futures_prices(
        futures_series=futures_series,
        start_date=start_date,
        constant_maturities_in_days=constant_maturities_in_days,
        price_field=price_field,
        spot_prices=spot_prices,
        volatilities=volatilities
    )


def update_generic_futures_prices(start_date=None,
                                  end_date=None,
                                  data_source_name='quandl'):

    date = start_date
    if date is None:
        date = dt.datetime.today() - BDay()

    fm = QuandlApi.get_futures_metadata()

    for i in range(0, len(fm)):

        futures_series = fm.loc[i, 'futures_series']
        dataset = fm.loc[i, 'generic_dataset']
        exchange = fm.loc[i, 'contracts_dataset']
        max_contract = int(fm.loc[i, 'num_generic_contracts'])
        constant_maturities_in_days = [0, 5, 10]
        for j in range(1, max_contract + 1):
            constant_maturities_in_days.append(
                constants.trading_days_per_month * j)

        update_generic_futures_prices_by_series(
            futures_series=futures_series,
            exchange_code=exchange,
            start_date=date,
            constant_maturities_in_days=constant_maturities_in_days)


def update_futures_prices_by_series(dataset=None,
                                    futures_series=None,
                                    exchange_code=None,
                                    source_series=None,
                                    contract_range=np.arange(1, 10),
                                    date=None):

    if source_series is None:
        source_series = futures_series

    # Get updated futures data
    futures_data = qfl_data.QuandlApi.update_daily_futures_prices(
        start_date=date,
        dataset=dataset,
        futures_series=source_series,
        contract_range=contract_range
    )

    # Map to appropriate contact
    series_data = db.get_futures_series(futures_series=futures_series,
                                        exchange_code=exchange_code)
    series_id = series_data['id'].values[0]

    where_str = " series_id = " + str(series_id) \
                + " and maturity_date >= '{0}'".format(dt.datetime.today())

    futures_contracts_data = db.get_data(
        table_name='futures_contracts',
        where_str=where_str)

    futures_data = futures_data.reset_index()

    df = pd.merge(left=futures_contracts_data[['id', 'ticker']],
                  right=futures_data,
                  on='ticker')
    cols = ['id', 'date', 'close_price', 'high_price', 'low_price',
            'open_price', 'settle_price', 'volume', 'open_interest']

    cols = list(set(cols).intersection(set(df.columns)))

    df = df[cols]

    futures_prices_table = db.get_table('futures_prices')

    if len(df) > 0:
        db.execute_db_save(df=df,
                           table=futures_prices_table,
                           extra_careful=False)

        precompute_historical_futures_days_to_maturity(
            futures_series=futures_series,
            start_date=date
        )
    else:
        x=1

    # TODO: what to do if we find a contract that we don't have in the DB


"""
-------------------------------------------------------------------------------
STRATEGIES
-------------------------------------------------------------------------------
"""


def ingest_strategy_backtest_data(start_date=None, **kwargs):

    sm = mm.initialize_strategy_environment(**kwargs)

    _ingest_strategy_backtest_data(sm=sm, start_date=start_date, **kwargs)


def _ingest_strategy_backtest_data(sm=None,
                                   start_date=default_start_date,
                                   **kwargs):

    # TODO: implement start date for this

    # Primary strategy versions
    strategy_versions = kwargs.get('strategy_versions',
                                   sm.get_strategy_versions())

    # Standard strategies
    standard_strategy_names = strategy_versions['standard'].keys()

    for strategy_name in standard_strategy_names:

        model = sm.strategies[strategy_name]

        for strategy_version in strategy_versions['standard'][strategy_name]:

            print(model.name + ' ' + strategy_version)

            model.settings = strategy_versions['standard']\
                [strategy_name][strategy_version]

            signal_data = sm.outputs[strategy_name]['signal_data']
            signal_data_z = sm.outputs[strategy_name]['signal_data_z']
            signal_pnl = sm.outputs[strategy_name]['signal_pnl']

            # These need to be done
            mm.archive_model_param_config(model_name=model.name,
                                          settings=model.settings)
            mm.archive_strategy_signals(model=model,
                                        signal_data=signal_data)

            # This is the signals and Z-scores, at the signal/strategy level
            mm.archive_strategy_signal_data(model=model,
                                            signal_data=signal_data,
                                            signal_data_z=signal_data_z)

            # This is the signal PNL, at the signal/strategy level
            mm.archive_strategy_signal_pnl(model=model,
                                           signal_pnl=signal_pnl)

            # This is the final PNL, at the strategy level
            mm.archive_model_outputs(model=model,
                                     outputs=sm.outputs[strategy_name])

            # TODO: shouldn't I archive the weights somehow?
            # Maybe do something like monthly rolling weights updates?

    # Portfolio strategies
    portfolio_strategy_names = strategy_versions['portfolio'].keys()

    for strategy_name in portfolio_strategy_names:

        model = sm.strategies[strategy_name]

        for strategy_version in strategy_versions['portfolio'][strategy_name]:

            model.settings = strategy_versions['portfolio']\
                [strategy_name][strategy_version]

            signal_data = sm.outputs[strategy_name]['signal_data']
            signal_data_z = sm.outputs[strategy_name]['signal_data_z']

            # These need to be done
            mm.archive_model_param_config(model_name=model.name,
                                          settings=model.settings)
            mm.archive_strategy_signals(model=model,
                                        signal_data=signal_data)

            # This is the signal PNL, at the signal/strategy level
            mm.compute_and_archive_portfolio_strategy_signal_pnl(
                model=model,
                signal_data=signal_data,
                backtest_update_start_date=start_date)

            # This is the signals and Z-scores, at the signal/underlying level
            mm.archive_portfolio_strategy_signal_data(
                model=model,
                signal_data=signal_data,
                signal_data_z=signal_data_z,
                backtest_update_start_date=start_date)

            # Retrieve signal PNL
            signal_pnl = mm.get_strategy_signal_data(model=model,
                                                     ref_entity_ids=['strategy',
                                                                     't'])
            signal_pnl = signal_pnl[['date', 'signal_name', 'pnl']] \
                .sort_values(['signal_name', 'date']) \
                .set_index(['date', 'signal_name'])['pnl'] \
                .unstack('signal_name')

            # We are not going to include all the categories of signals
            included_signals = ['iv_10', 'iv_21', 'iv_42', 'iv_63', 'iv_126',
                                'iv_252',
                                'rv_iv_10', 'rv_iv_21', 'rv_iv_42', 'rv_iv_63',
                                'rv_iv_126', 'rv_iv_252',
                                'rv_10', 'rv_21', 'rv_42', 'rv_63', 'rv_126',
                                'ts_0', 'ts_1', 'ts_5', 'ts_10', 'ts_21']

            # Run the optimization and the final backtest
            positions, portfolio_summary, sec_master, optim_output = \
                model.compute_master_backtest(
                    signal_pnl=signal_pnl,
                    signal_data=signal_data,
                    included_signals=included_signals,
                    backtest_update_start_date=start_date
                )

            # Archive the portfolio summary (strategy-level PNL)
            mm.archive_model_outputs(model=model, outputs=portfolio_summary)

            # Archive the security master
            mm.archive_portfolio_strategy_security_master(model=model,
                                                          sec_master=sec_master)

            # Archive the positions
            mm.archive_portfolio_strategy_positions(model=model,
                                                    positions=positions)


