"""
This module should ideally be ONLY tasked with the automation and triggering of
ETL tasks and should rely on other modules to actually implement those tasks?
EG, we shouldn't need references here to YahooApi...
"""

import pandas as pd
import datetime as dt
import numpy as np
from pandas.tseries.offsets import BDay
import urllib

import qfl.core.data_interfaces as qfl_data
from qfl.core.data_interfaces import YahooApi, QuandlApi, FigiApi, DataScraper
from qfl.core.data_interfaces import DatabaseInterface as db
import qfl.core.utils as utils
import qfl.core.constants as constants
import logging

# Default start date for historical data ingests
default_start_date = dt.datetime(1990, 1, 1)
default_equity_indices = ['SPX', 'NDX', 'UKX']


def initialize_data_environment():
    db.initialize()


def test_airflow():

    df = pd.DataFrame(np.random.randn(10, 5))
    print('successfully ran airflow test...')
    return True


def test_airflow_awesome():

    df = pd.DataFrame(np.random.randn(10, 5))
    # df.to_csv('test' + dt.datetime.today().__str__() + '.csv')
    print('test CSV is printing...')
    return True


def daily_futures_price_ingest():

    initialize_data_environment()
    data_source = 'quandl'
    update_futures_prices(_db=db)


def daily_equity_price_ingest():

    initialize_data_environment()
    data_source = 'yahoo'

    # Prep
    equities, equity_prices_table, ids, equity_tickers, rows = \
        _prep_equity_price_load(ids=None, _db=db)

    update_equity_prices(ids=ids,
                         data_source=data_source,
                         _db=db)


def historical_equity_price_ingest():

    initialize_data_environment()
    date = utils.closest_business_day()
    data_source = 'yahoo'
    load_historical_equity_prices(ids=None,
                                  start_date=default_start_date,
                                  end_date=date,
                                  data_source=data_source,
                                  _db=db)


def historical_dividends_ingest():

    initialize_data_environment()
    load_historical_dividends(ids=None,
                              start_date=default_start_date,
                              end_date=dt.datetime.today(),
                              data_source='yahoo')


def _prep_equity_price_load(ids=None,
                            _db=None,
                            equity_indices=default_equity_indices):

    s = "select * from equities e where id in " \
        "(select equity_id from equity_index_members where index_id in " \
        "(select index_id from equity_indices where ticker in {0}))" \
        .format(tuple(equity_indices))

    equities = _db.read_sql(s)
    equities.index = equities['id']

    equity_prices_table = _db.get_table(table_name='equity_prices',)

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


def _update_option_attrs(raw_data=None, _db=None):

    options_table = _db.get_table(table_name='equity_options')
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
    option_attributes['underlying_id'] = _db.get_equity_ids(
        equity_tickers=raw_data['Underlying'])

    _db.execute_db_save(df=option_attributes,
                        table=options_table,
                        use_column_as_key='ticker')

    # Get their ID's
    t = tuple([str(ticker) for ticker in option_attributes['ticker']])
    q = 'select ticker, id from equity_options where ticker in {0}'.format(t)
    ticker_id_map = _db.read_sql(query=q)
    ticker_id_map.index = ticker_id_map['ticker']

    return ticker_id_map, option_attributes['ticker']


def update_option_prices(ids=None,
                         data_source='yahoo',
                         _db=None):
    for id_ in ids:
        update_option_prices_one(id_=id_,
                                 data_source=data_source,
                                 _db=_db)


def update_option_prices_one(id_=None,
                             data_source='yahoo',
                             _db=None):

    ticker = _db.get_equity_tickers(ids=[id_])[0]

    if data_source == 'yahoo':

        # Get raw data
        tmp = YahooApi.retrieve_options_data(ticker)
        raw_data = tmp[0]

        # Update option universe
        ticker_id_map, tickers = _update_option_attrs(raw_data, _db)

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
                         data_source='yahoo',
                         _db=None):

    date = dt.datetime.today() - BDay()

    load_historical_equity_prices(ids=ids,
                                  start_date=date,
                                  end_date=date,
                                  data_source=data_source,
                                  _db=_db,
                                  batch_ones=True)


def _load_historical_equity_prices(ids=None,
                                   start_date=default_start_date,
                                   end_date=dt.datetime.today(),
                                   data_source='yahoo',
                                   _db=None,
                                   batch_size=100):
    # Prep
    equities, equity_prices_table, ids, equity_tickers, rows = \
        _prep_equity_price_load(ids, _db)

    if data_source == 'yahoo':

        # TODO: move this to a sensible place
        equity_tickers = [ticker.replace(".LN", ".L")
                          for ticker in equity_tickers]

        big_prices_df = None
        num_batches = int(np.ceil(float(len(ids)) / batch_size))

        for i in range(0, num_batches):

            # logging.info("batch " + str(i) + " + out of " + str(num_batches))

            lrange = i * batch_size
            urange = np.min([(i + 1) * batch_size, len(ids)-1])

            if urange == 0:
                prices = YahooApi.retrieve_prices(equity_tickers,
                                                  start_date, end_date)
            else:
                prices = YahooApi.retrieve_prices(equity_tickers[lrange:urange],
                                                  start_date, end_date)

            if isinstance(prices, pd.Panel):
                prices_df = prices.to_frame()
            else:
                prices_df = prices

            yahoo_fields = ['id', 'date', 'Open', 'High', 'Low',
                            'Close', 'Volume', 'Adj Close']

            db_fields = ['id', 'date', 'open_price', 'high_price', 'low_price',
                         'last_price', 'volume', 'adj_close']

            # Remove indices to prepare for database
            prices_df.index.names = ['date', 'ticker']
            prices_df = prices_df.reset_index()

            if big_prices_df is None:
                big_prices_df = prices_df.copy(deep=True)
            else:
                big_prices_df = big_prices_df.append(prices_df)

        # Merge with ID's
        mapped_prices = pd.merge(left=big_prices_df,
                                 right=rows,
                                 on='ticker',
                                 how='inner')

    else:
        raise NotImplementedError

    # Map to database column structure
    equity_prices_data = pd.DataFrame(index=mapped_prices.index,
                                      columns=equity_prices_table.columns.keys())
    for i in range(0, len(yahoo_fields)):
        equity_prices_data[db_fields[i]] = mapped_prices[yahoo_fields[i]]

    logging.info("archiving " + str(len(equity_prices_data)) + " equity prices...")

    equity_prices_data = equity_prices_data.reset_index()
    _db.execute_db_save(equity_prices_data, equity_prices_table)


def load_historical_equity_prices(ids=None,
                                  start_date=default_start_date,
                                  end_date=dt.datetime.today(),
                                  data_source='yahoo',
                                  _db=None,
                                  batch_ones=True):

    if batch_ones:
        for _id in ids:

            try:
                _load_historical_equity_prices([_id],
                                               start_date,
                                               end_date,
                                               data_source,
                                               _db)
                logging.info('ran for ' + str(_id))
            except:
                logging.info('failed for ' + str(_id))
    else:
        _load_historical_equity_prices(ids,
                                       start_date,
                                       end_date,
                                       data_source,
                                       _db)


def load_historical_dividends(ids=None,
                              start_date=None,
                              end_date=None,
                              data_source='yahoo',
                              _db=None):

    if end_date is None:
        end_date = dt.datetime.today()

    if start_date is None:
        start_date = default_start_date

    # Use existing routine to get tickers and id defaults
    equities, equity_prices_table, ids, tickers, rows = \
        _prep_equity_price_load(ids)

    if data_source == 'yahoo':

        dividends = YahooApi.retrieve_dividends(equity_tickers=tickers,
                                                start_date=start_date,
                                                end_date=end_date)

        schedules_table = _db.get_table(table_name='equity_schedules')
        equities = _db.get_data(table_name='equities')

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

    _db.execute_db_save(df=schedules_data, table=schedules_table)



def add_equities_from_index_web():

    url_root = "http://www.dax-indices.com/MediaLibrary/Document/WeightingFiles/08/"
    url = url_root + "DAX_ICR.20160805.xls"
    file = urllib.urlretrieve(url, "data/dax.xls")
    tmp = pd.read_csv("data/dax.csv")

    import urllib
    url = "http://cfe.cboe.com/data/DailyVXFuturesEODValues/DownloadFS.aspx"
    file = urllib.urlretrieve(url, "data/vix_settle.csv")
    tmp = pd.read_csv("data/vix_settle.csv")


def add_equities_from_index(ticker=None,
                            country=None,
                            method='quandl',
                            _db=None):

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
    add_equities_from_list(tickers=tickers, country=country, _db=_db)

    # Get the equities we just created
    where_str = " ticker in {0}".format(tuple(tickers))
    equities_table_data = _db.get_data(table_name='equities',
                                       where_str=where_str)

    # Find the index mapping
    indices = _db.get_data(table_name='equity_indices')
    index_id = indices[indices['ticker'] == ticker]['index_id'].values[0]

    # Get index members table
    index_members_table = _db.get_table(table_name='equity_index_members')
    index_membership_data = pd.DataFrame(
        columns=index_members_table.columns.keys())
    index_membership_data['equity_id'] = equities_table_data['id']
    index_membership_data['valid_date'] = dt.date.today()
    index_membership_data['index_id'] = index_id

    # Update equity index membership table
    _db.execute_db_save(df=index_membership_data, table=index_members_table)


def add_equities_from_list(tickers=None, country=None, _db=None):

    df = pd.DataFrame(data=tickers, columns=['ticker'])
    df['country'] = country
    equities_table = _db.get_table(table_name='equities')
    _db.execute_db_save(df=df,
                        table=equities_table,
                        extra_careful=False,
                        use_column_as_key='ticker')


def add_volatility_futures_series(_db=None):

    # Manual because doesn't exist in Quandl source
    futures_series_table = _db.get_table('futures_series')

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


def add_futures_series(_db=None):

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
    futures_series_table = _db.get_table('futures_series')
    for col in futures_series.columns:
        if col not in futures_series_table.columns.keys():
            del futures_series[col]

    # Unique (contracts, not session times)
    futures_series = futures_series.drop_duplicates(
        subset=['series', 'exchange'])

    _db.execute_db_save(df=futures_series, table=futures_series_table)


def load_historical_generic_futures_prices(_db=None,
                                           futures_series=None,
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

    _db.execute_db_save(df=contracts_retrieved,
                        table=futures_contracts_table,
                        use_column_as_key='ticker',
                        extra_careful=False)

    # Prices
    futures_prices_table = _db.get_table(table_name='generic_futures_prices')
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
    futures_contracts = _db.get_data(table_name='generic_futures_contracts')

    tmp = pd.merge(left=futures_contracts[['id', 'ticker']],
                   right=futures_data,
                   on='ticker')

    for col in tmp.columns:
        if col not in futures_prices_table.columns.keys():
            del tmp[col]

    result = _db.execute_bulk_insert(df=tmp, table=futures_prices_table)
    return result


def load_historical_generic_vix_futures_prices(_db=None):

    dataset = 'CHRIS'
    futures_series = 'VX'
    source_series = 'CBOE_VX'

    load_historical_generic_futures_prices(_db=_db,
                                           dataset=dataset,
                                           futures_series=futures_series,
                                           source_series=source_series)


def load_historical_vix_futures_prices(_db=None):

    dataset = 'CBOE'
    futures_series = 'VX'
    start_date = dt.datetime(2007, 3, 24)

    load_historical_futures_prices(_db=db,
                                   dataset=dataset,
                                   futures_series=futures_series,
                                   start_date=start_date)


def load_historical_futures_prices(_db=None,
                                   dataset=None,
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
    futures_series_data = _db.get_data(table_name='futures_series')
    ind = futures_series_data.index[
        futures_series_data['series'] == futures_series]
    series_id = futures_series_data.loc[ind, 'id'].values[0]
    futures_contracts_retrieved['series_id'] = series_id

    # Futures table
    futures_contracts_table = _db.get_table(table_name='futures_contracts')

    # Archive
    _db.execute_db_save(df=futures_contracts_retrieved,
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
                          maturity_dates=None,
                          _db=None):

    futures_contracts_table = _db.get_table('futures_contracts')
    futures_series_data = _db.get_futures_series(futures_series)
    series_id = futures_series_data['id'].values[0]

    df = pd.DataFrame(columns=['series_id', 'ticker', 'maturity_date'],
                      index=tickers)

    df['ticker'] = tickers
    df['maturity_date'] = maturity_dates
    df['series_id'] = series_id

    _db.execute_db_save(df=df,
                        table=futures_contracts_table,
                        use_column_as_key='ticker')


def update_futures_prices(_db=None):

    date = dt.datetime.today() - BDay()

    # VIX futures
    futures_series = 'VX'
    source_series = 'VX'
    dataset = 'CBOE'
    contract_range = np.arange(1, 10)
    update_futures_prices_by_series(_db=_db,
                                    dataset=dataset,
                                    futures_series=futures_series,
                                    source_series=source_series,
                                    contract_range=contract_range,
                                    date=date)

    # V2X futures
    futures_series = 'FVS'
    source_series = 'FVS'
    dataset = 'EUREX'
    contract_range = np.arange(1, 8)
    update_futures_prices_by_series(_db=_db,
                                    dataset=dataset,
                                    futures_series=futures_series,
                                    source_series=source_series,
                                    contract_range=contract_range,
                                    date=date)


def update_futures_prices_by_series(_db=None,
                                    dataset=None,
                                    futures_series=None,
                                    source_series=None,
                                    contract_range=np.arange(1, 10),
                                    date=dt.datetime.today()):

    if source_series is None:
        source_series = futures_series

    # Get updated futures data
    futures_data = qfl_data.QuandlApi.update_futures_prices(
        date=date,
        dataset=dataset,
        futures_series=source_series,
        contract_range=contract_range
    )

    # Map to appropriate contact
    series_data = _db.get_futures_series(futures_series=futures_series)
    series_id = series_data['id'].values[0]

    where_str = " series_id = " + str(series_id) \
                + " and maturity_date >= '" \
                + dt.datetime.today().date().__str__() + "'"

    futures_contracts_data = _db.get_data(
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

    futures_prices_table = _db.get_table('futures_prices')
    _db.execute_db_save(df=df,
                        table=futures_prices_table,
                        extra_careful=False)


    # TODO: what to do if we find a contract that we don't have in the DB
    # Need a way to automatically get maturity data... barchart has this

    # # Futures table
    # futures_contracts_table = _db.get_table(table_name='futures_contracts')
    #
    # # Archive
    # _db.execute_db_save(df=futures_contracts,
    #                     table=futures_contracts_table,
    #                     use_column_as_key='ticker')


def update_vix_futures_settle_prices(overwrite_date=None, _db=None):

    prices = DataScraper.update_vix_settle_price(overwrite_date)

