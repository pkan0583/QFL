"""
This module should ideally be ONLY tasked with the automation and triggering of
ETL tasks and should rely on other modules to actually implement those tasks?
EG, we shouldn't need references here to YahooApi...
"""
import os
import pandas as pd
import datetime as dt
import numpy as np
from pandas.tseries.offsets import BDay
import urllib, urllib2
import logging

import qfl.core.data_interfaces as qfl_data
import qfl.utilities.basic_utilities as utils
from qfl.core.data_interfaces import YahooApi, QuandlApi, FigiApi, DataScraper
from qfl.core.database_interface import DatabaseInterface as db
import qfl.core.constants as constants
import qfl.core.market_data as md

# Default start date for historical data ingests
default_start_date = dt.datetime(1990, 1, 1)
default_equity_indices = ['SPX', 'NDX', 'UKX']
db.initialize()


"""
-------------------------------------------------------------------------------
DATA INGEST: ENTRY POINTS
-------------------------------------------------------------------------------
"""


def _get_execution_date(date=None, **kwargs):
    airflow_date = kwargs.get('execution_date', None)
    if airflow_date is not None:
        date = airflow_date
    elif date is None:
        date = utils.workday(dt.datetime.today(), num_days=-1)
    return date


def daily_equity_index_price_ingest(date=None, **kwargs):

    date = _get_execution_date(date, **kwargs)

    logging.info("starting daily equity price ingest for "
                 + str(date) + "...")

    data_source_name = 'yahoo'
    ingest_historical_index_prices(start_date=date,
                                   end_date=date,
                                   data_source_name=data_source_name)

    logging.info("completed daily futures price ingest!")


def daily_futures_price_ingest(date=None, **kwargs):

    date = _get_execution_date(date, **kwargs)

    logging.info("starting daily futures price ingest "
                + str(date) + "...")

    data_source_name = 'quandl'
    update_futures_prices(date=date)

    logging.info("completed daily futures price ingest!")


def test_process(date=None, **kwargs):

    date = _get_execution_date(date, **kwargs)
    logging.info("testing execution date..." + str(date))


def daily_generic_futures_price_ingest(date=None, **kwargs):

    date = _get_execution_date(date, **kwargs)

    logging.info("starting daily generic futures price ingest for "
                 + str(date) + "...")

    data_source_name = 'quandl'
    update_generic_futures_prices(date=date)

    logging.info("completed daily generic futures price ingest!")


def daily_equity_price_ingest(date=None, **kwargs):

    date = _get_execution_date(date, **kwargs)

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
DATA INGEST: DETAILS
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


def update_equity_prices(ids=None,
                         data_source_name='yahoo',
                         date=None,
                         batch_size=1):

    if date is None:
        date = utils.workday(num_days=-1)

    ingest_historical_equity_prices(ids=ids,
                                    start_date=date,
                                    end_date=date,
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

    db.execute_db_save(df=prices_df.head(),
                       table=index_prices_table,
                       extra_careful=False,
                       time_series=True)


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

            try:
                ingest_historical_equity_prices(ids=ids[lrange:urange],
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

        mapping = FigiApi.retrieve_mapping(id_type='TICKER',
                                           ids=tickers,
                                           exch_codes=exchange_codes)

        mapping['security_type'] = security_type

        cols = ['figi_id', 'composite_figi_id', 'bbg_sector', 'exchange_code',
                'security_type', 'security_sub_type', 'ticker', 'name']

        mapping = mapping[cols]

        db.execute_db_save(df=mapping,
                           table=securities_table,
                           extra_careful=False)


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
                   'Delivery Months': 'delivery months'
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

    db.execute_db_save(df=futures_series, table=futures_series_table)


def ingest_historical_generic_futures_prices(futures_series=None,
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
    futures_data.index.names = ['ticker', 'date']
    futures_data = futures_data.reset_index()

    # Contracts
    futures_contracts_table = db.get_table(
        table_name='generic_futures_contracts')
    contracts_retrieved = pd.DataFrame(futures_data.groupby('ticker')
                                       .last()['contract_number'])
    futures_series_data = db.get_data(table_name='futures_series')
    ind = futures_series_data.index[
        futures_series_data['series'] == futures_series]
    series_id = futures_series_data.loc[ind, 'id'].values[0]
    contracts_retrieved['series_id'] = series_id
    contracts_retrieved = contracts_retrieved.reset_index()

    db.execute_db_save(df=contracts_retrieved,
                       table=futures_contracts_table,
                       use_column_as_key='ticker',
                       extra_careful=False)

    # Prices
    futures_prices_table = db.get_table(table_name='generic_futures_prices')
    futures_data['open_interest'] = \
        futures_data['Prev. Day Open Interest'].shift(-1)
    rename_cols = {'Close': 'close_price',
                   'High': 'high_price',
                   'Low': 'low_price',
                   'Open': 'open_price',
                   'Settle': 'settle_price',
                   'Total Volume': 'volume'}
    futures_data = futures_data.rename(columns=rename_cols)

    # Filter for valid prices
    futures_data = futures_data[
        futures_data['settle_price'] > 0]

    # Join with id
    futures_contracts = db.get_data(table_name='generic_futures_contracts')

    tmp = pd.merge(left=futures_contracts[['id', 'ticker']],
                   right=futures_data,
                   on='ticker')

    for col in tmp.columns:
        if col not in futures_prices_table.columns.keys():
            del tmp[col]

    tmp = tmp.fillna(value=constants.invalid_value)

    result = db.execute_db_save(df=tmp,
                                table=futures_prices_table,
                                extra_careful=False,
                                time_series=True)
    return result


def ingest_historical_futures_days_to_maturity(futures_series=None,
                                               start_date=None,
                                               generic=False):

    # Retreive the basic data
    if generic:
        table_name = 'generic_futures_prices'
        futures_prices = md.get_generic_futures_prices_from_series(
            futures_series=futures_series,
            start_date=start_date,
            include_contract_map=True)
    else:
        table_name = 'futures_prices'
        futures_prices = md.get_futures_prices_from_series(
            futures_series=futures_series,
            start_date=start_date)

    # Retrieve the series and the holiday calendar
    series_data = db.get_data(table_name='futures_series',
                              where_str=" series = '" + futures_series + "'")
    series_exchange = series_data['exchange'].values[0]
    calendar_name = utils.DateUtils.exchange_calendar_map[series_exchange]

    # Calculate net workdays
    start_dates = futures_prices.index.get_level_values('date')
    end_dates = futures_prices['maturity_date'].values
    futures_prices['days_to_maturity'] = utils.networkdays(
        start_date=start_dates,
        end_date=end_dates,
        calendar_name=calendar_name
    )

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


def load_historical_generic_vix_futures_prices():

    dataset = 'CHRIS'
    futures_series = 'VX'
    source_series = 'CBOE_VX'

    ingest_historical_generic_futures_prices(dataset=dataset,
                                             futures_series=futures_series,
                                             source_series=source_series)


def load_historical_vix_futures_prices():

    dataset = 'CBOE'
    futures_series = 'VX'
    start_date = dt.datetime(2007, 3, 24)

    load_historical_futures_prices(dataset=dataset,
                                   futures_series=futures_series,
                                   start_date=start_date)


def load_historical_futures_prices(dataset=None,
                                   futures_series=None,
                                   start_date=None):

    # Need logic to identify data sources etc
    # Also the transform below could ideally be moved

    # Get the big dataset
    futures_data = qfl_data.QuandlApi.retrieve_historical_futures_prices(
        start_date=start_date,
        futures_series=futures_series,
        dataset=dataset
    )
    futures_data.index.names = ['ticker', 'date']

    # Update our contract table

    # Get expiration dates: note this only works for expired contracts
    futures_contracts_retrieved = futures_data.reset_index() \
        .groupby('ticker') \
        .last()['date']
    futures_contracts_retrieved = pd.DataFrame(futures_contracts_retrieved).reset_index()
    futures_contracts_retrieved = futures_contracts_retrieved.rename(
        columns={'date': 'maturity_date'})

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

    # Prices
    futures_prices_table = db.get_table(table_name='futures_prices')
    futures_data = futures_data.reset_index()

    if 'Prev. Day Open Interest' in futures_data.columns:
        futures_data['open_interest'] = \
            futures_data['Prev. Day Open Interest'].shift(-1)
    rename_cols = {'Close': 'close_price',
                   'High': 'high_price',
                   'Low': 'low_price',
                   'Open': 'open_price',
                   'Settle': 'settle_price',
                   'Total Volume': 'volume'}
    futures_data = futures_data.rename(columns=rename_cols)

    # Filter for valid prices
    futures_data = futures_data[
        futures_data['settle_price'] > 0]

    # Join with id
    futures_contracts = db.get_data(table_name='futures_contracts')

    tmp = pd.merge(left=futures_contracts[['id', 'ticker']],
                   right=futures_data,
                   on='ticker')

    for col in tmp.columns:
        if col not in futures_prices_table.columns.keys():
            del tmp[col]

    result = db.execute_bulk_insert(df=tmp, table=futures_prices_table)

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


def update_futures_prices(date=None):

    if date is None:
        date = dt.datetime.today() - BDay()

    # VIX futures
    futures_series = 'VX'
    source_series = 'VX'
    dataset = 'CBOE'
    contract_range = np.arange(1, 10)
    update_futures_prices_by_series(dataset=dataset,
                                    futures_series=futures_series,
                                    source_series=source_series,
                                    contract_range=contract_range,
                                    date=date)

    # V2X futures
    futures_series = 'FVS'
    source_series = 'FVS'
    dataset = 'EUREX'
    contract_range = np.arange(1, 8)
    update_futures_prices_by_series(dataset=dataset,
                                    futures_series=futures_series,
                                    source_series=source_series,
                                    contract_range=contract_range,
                                    date=date)


def update_generic_futures_prices(date=None):

    # VIX futures
    futures_series = 'VX'
    source_series = 'CBOE_VX'
    dataset = 'CHRIS'
    contract_range = np.arange(1, 10)

    ingest_historical_generic_futures_prices(
        futures_series=futures_series,
        source_series=source_series,
        contract_range=contract_range,
        dataset=dataset,
        start_date=date)

    load_historical_generic_futures_days_to_maturity(
        futures_series=futures_series,
        start_date=date
    )

    # V2X futures
    futures_series = 'FVS'
    source_series = 'EUREX_FVS'
    dataset = 'CHRIS'
    contract_range = np.arange(1, 8)

    ingest_historical_generic_futures_prices(
        futures_series=futures_series,
        source_series=source_series,
        contract_range=contract_range,
        dataset=dataset,
        start_date=date)

    load_historical_generic_futures_days_to_maturity(
        futures_series=futures_series,
        start_date=date
    )


def update_futures_prices_by_series(dataset=None,
                                    futures_series=None,
                                    source_series=None,
                                    contract_range=np.arange(1, 10),
                                    date=None):

    if source_series is None:
        source_series = futures_series

    # Get updated futures data
    futures_data = qfl_data.QuandlApi.update_daily_futures_prices(
        date=date,
        dataset=dataset,
        futures_series=source_series,
        contract_range=contract_range
    )

    # Map to appropriate contact
    series_data = db.get_futures_series(futures_series=futures_series)
    series_id = series_data['id'].values[0]

    where_str = " series_id = " + str(series_id) \
                + " and maturity_date >= '" \
                + dt.datetime.today().date().__str__() + "'"

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
    db.execute_db_save(df=df,
                       table=futures_prices_table,
                       extra_careful=False)

    ingest_historical_futures_days_to_maturity(
        futures_series=futures_series,
        start_date=date
    )

    # TODO: what to do if we find a contract that we don't have in the DB



def update_vix_futures_settle_prices(overwrite_date=None):

    prices = DataScraper.update_vix_settle_price(overwrite_date)

