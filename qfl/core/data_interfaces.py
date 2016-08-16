import os
import pandas as pd
import datetime as dt
import pandas_datareader.data as pdata
from pandas.tseries.offsets import BDay
import quandl as ql
import numpy as np
import sqlalchemy as sa
import urllib
import dateutil
import time
from sqlalchemy.sql.expression import bindparam
from bs4 import BeautifulSoup
import requests
import logging

import urllib2
import simplejson as json

import qfl.core.constants as constants

from ib.ext.Contract import Contract
from ib.ext.Order import Order
from ib.opt import Connection, message
from ib.ext.ContractDetails import ContractDetails

import qfl.utilities.basic_utilities as utils

'''
--------------------------------------------------------------------------------
INTERACTIVE BROKERS API
--------------------------------------------------------------------------------
'''


class IBApi(object):

    ib_username = os.getenv('IB_USERNAME')
    client_code = 100
    port_num = 7496
    tws_conn = None
    account_value = None
    portfolio = None

    # These guys are lists of messages passed by the API
    price_updates = None
    option_computation_updates = None
    historical_prices = None

    # Need to adapt these to multiple active requests
    active_data_symbol = None
    data_wait = False

    bar_length_options = (
        '1 secs',
        '5 secs',
        '10 secs',
        '15 secs',
        '30 secs',
        '1 min',
        '2 mins',
        '3 mins',
        '5 mins',
        '10 mins',
        '15 mins',
        '20 mins',
        '30 mins',
        '1 hour',
        '2 hours',
        '3 hours',
        '4 hours',
        '8 hours',
        '1 day',
        '1 W',
        '1 M'
    )

    tick_fields = {
        0: 'bid_size',
        1: 'bid_price',
        2: 'ask_price',
        3: 'ask_size',
        4: 'last_price',
        5: 'last_size',
        6: 'high_price',
        7: 'low_price',
        8: 'volume',
        9: 'close_price',
        10: 'bid_option_computation',
        11: 'ask_option_computation',
        12: 'last_option_computation',
        13: 'model_option_computation',
        14: 'open_tick',
        22: 'open_interest',
        23: 'option_historical_vol',
        24: 'option_implied_vol',
        25: 'option_bid_exch',
        26: 'option_ask_exch',
        27: 'option_call_open_interest',
        28: 'option_put_open_interest',
        29: 'option_call_volume',
        30: 'option_put_volume',
        45: 'last_timestamp'
    }

    @staticmethod
    def gen_tick_id():
        from random import randint
        i = randint(1, 1000)
        # while True:
        #     yield i
        #     i += 1
        return i

    @staticmethod
    def error_handler(msg):
        """Handles the capturing of error messages"""
        print "Server Error: %s" % msg

    @staticmethod
    def reply_handler(msg):
        """Handles of server replies"""
        print "Server Response: %s, %s" % (msg.typeName, msg)

    def portfolio_update_handler(self, msg):
        """Handles portfolio updates"""
        if self.portfolio is None:
            self.portfolio = list()
        self.portfolio.append(msg)

    def account_value_update_handler(self, msg):
        """Handles account updates"""
        if self.account_value is None:
            self.account_value = list()
        self.account_value.append(msg)

    def market_data_handler(self, msg):
        """Handles price updates"""
        if self.price_updates is None:
            self.price_updates = list()
        if msg.price == -1.0:
            self.data_wait = False
            x = 1
        else:
            self.price_updates.append(msg)

    def option_computation_handler(self, msg):
        """Handles option computation updates"""
        if self.option_computation_updates is None:
            self.option_computation_updates = list()
        self.option_computation_updates.append(msg)

    def historical_market_data_handler(self, msg):
        """ handles historical data updates """
        if self.historical_prices is None:
            self.historical_prices = list()
        if msg.open == -1 or msg.date.__contains__('finished'):
            self.tws_conn.cancelHistoricalData(msg.tickerId)
            self.data_wait = False
        else:
            self.historical_prices.append(msg)

    @staticmethod
    def create_contract(symbol, sec_type, exch, prim_exch, curr):
        """Create a Contract object defining what will
        be purchased, at which exchange and in which currency.

        symbol - The ticker symbol for the contract
        sec_type - The security type for the contract ('STK' is 'stock')
        exch - The exchange to carry out the contract on
        prim_exch - The primary exchange to carry out the contract on
        curr - The currency in which to purchase the contract"""
        contract = Contract()
        contract.m_symbol = symbol
        contract.m_secType = sec_type
        contract.m_exchange = exch
        contract.m_primaryExch = prim_exch
        contract.m_currency = curr
        return contract

    @staticmethod
    def create_order(order_type, quantity, action):
        """Create an Order object (Market/Limit) to go long/short.

        order_type - 'MKT', 'LMT' for Market or Limit orders
        quantity - Integral number of assets to order
        action - 'BUY' or 'SELL'"""
        order = Order()
        order.m_orderType = order_type
        order.m_totalQuantity = quantity
        order.m_action = action
        return order

    def disconnect(self):
        return self.tws_conn.disconnect()

    def request_positions(self):
        self.tws_conn.reqPositions()

    def request_account_updates(self):
        self.tws_conn.reqAccountUpdates(True, self.ib_username)

    def initialize(self):

        self.tws_conn = Connection.create(port=self.port_num,
                                          clientId=self.client_code)
        self.tws_conn.registerAll(self.reply_handler)
        self.tws_conn.register(self.portfolio_update_handler,
                               message.updatePortfolio)
        self.tws_conn.register(self.account_value_update_handler,
                               message.updateAccountValue)
        self.tws_conn.register(self.market_data_handler,
                               message.tickPrice)
        self.tws_conn.register(self.option_computation_handler,
                               message.tickOptionComputation)
        self.tws_conn.register(self.historical_market_data_handler,
                               message.historicalData)
        self.tws_conn.connect()

    # def connect(self):
        # Connect to the Trader Workstation (TWS) running on the
        # usual port of 7496, with a clientId of 100
        # (The clientId is chosen by us and we will need
        # separate IDs for both the execution connection and
        # market data connection)
        # tws_conn = Connection.create(port=self.port_num,
        #                              clientId=self.client_code)
        # tws_conn.connect()
        #
        # # Assign the error handling function defined above
        # # to the TWS connection
        # tws_conn.register(self.error_handler, 'Error')

        # Assign all of the server reply messages to the
        # reply_handler function defined above
        # tws_conn.registerAll(self.reply_handler)

        # # Create an order ID which is 'global' for this session. This
        # # will need incrementing once new orders are submitted.
        # order_id = 1
        #
        # # Create a contract in GOOG stock via SMART order routing
        # goog_contract = IBApi.create_contract('GOOG', 'STK', 'SMART', 'SMART', 'USD')
        #
        # # Go long 100 shares of Google
        # goog_order = IBApi.create_order('MKT', 100, 'BUY')
        #
        # # Use the connection to the send the order to IB
        # tws_conn.placeOrder(order_id, goog_contract, goog_order)
        #
        # # Disconnect from TWS
        # tws_conn.disconnect()

    def retrieve_historical_prices(self,
                                   ticker=None,
                                   currency=None,
                                   security_type=None,
                                   exchange=None,
                                   strike_price=None,
                                   maturity_date=None,
                                   option_type=None,
                                   multiplier=None,
                                   duration_str=None,
                                   bar_size=None):

        self.active_data_symbol = ticker

        px_contract = IBApi.create_contract(
            symbol=ticker,
            sec_type=security_type,
            exch=exchange,
            curr=currency,
            prim_exch=exchange
        )
        px_contract.m_strike = strike_price
        px_contract.m_expiry = maturity_date
        px_contract.m_right = option_type
        px_contract.m_multiplier = multiplier

        tick_id = 1
        end_time = time.strftime('%Y%m%d %H:%M:%S')

        # Get bids
        self.tws_conn.reqHistoricalData(tickerId=tick_id,
                                        contract=px_contract,
                                        endDateTime=end_time,
                                        durationStr=duration_str,
                                        barSizeSetting=bar_size,
                                        whatToShow='BID',
                                        useRTH=0,
                                        formatDate=1)

        self.data_wait = True
        i = 0
        while self.data_wait and i < 90:
            print(i)
            i += 1
            time.sleep(1)

        historical_data = self.process_historical_prices(
            self.historical_prices)

        raw_historical_data = list(self.historical_prices)
        self.historical_prices = None

        return historical_data, raw_historical_data

    def process_option_computations(self, option_computations):

        column_map = {'tickerId': 'ticker_id',
                      'field': 'field',
                      'impliedVol': 'implied_volatility',
                      'delta': 'delta',
                      'gamma': 'gamma',
                      'vega': 'vega',
                      'theta': 'theta',
                      'undPrice': 'underlying_price',
                      'pvDividend': 'pv_dividend',
                      'optPrice': 'price'}

        option_computations_df = pd.DataFrame(columns=column_map.values())

        for i in range(0, len(option_computations)):
            for col in column_map:
                data_i = dict(option_computations[i].items())
                option_computations_df.loc[i, column_map[col]] = data_i[col]

        option_computations_df['ticker'] = self.active_data_symbol

        return option_computations_df

    def process_historical_prices(self, historical_prices):

        column_map = {'date': 'date',
                      'open': 'open_price',
                      'close': 'close_price',
                      'low': 'low_price',
                      'high': 'high_price',
                      'volume': 'volume',
                      'WAP': 'vwap'}

        prices_df = pd.DataFrame(columns=column_map.values())

        for i in range(0, len(historical_prices)):
            for col in column_map:
                data_i = dict(historical_prices[i].items())
                prices_df.loc[i, column_map[col]] = data_i[col]

        prices_df['ticker'] = self.active_data_symbol

        return prices_df

    def process_prices(self, prices):

        column_map = {'tickerId': 'ticker_id',
                      'field': 'price_type',
                      'price': 'price'}

        prices_df = pd.DataFrame(columns=column_map.values())

        for i in range(0, len(prices)):
            for col in column_map:
                data_i = dict(prices[i].items())
                prices_df.loc[i, column_map[col]] = data_i[col]

        prices_df['ticker'] = self.active_data_symbol

        return prices_df

    def retrieve_prices(self,
                        underlying_ticker=None,
                        security_ticker=None,
                        currency=None,
                        security_type=None,
                        exchange=None,
                        strike_price=None,
                        maturity_date=None,
                        option_type=None,
                        multiplier=None,
                        subscribe=False):

        px_contract = IBApi.create_contract(
            symbol=underlying_ticker,
            sec_type=security_type,
            exch=exchange,
            curr=currency,
            prim_exch=exchange
        )
        px_contract.m_localSymbol = security_ticker
        px_contract.m_strike = strike_price
        px_contract.m_expiry = maturity_date
        px_contract.m_right = option_type
        px_contract.m_multiplier = multiplier

        ticker_id = self.gen_tick_id()

        self.tws_conn.reqMktData(ticker_id, px_contract, '', False)

        self.data_wait = True
        if not subscribe:
            i = 0
            while self.data_wait and i < 90:
                print(i)
                i += 1
                time.sleep(1)

            prices_df = self.process_prices(self.price_updates)
            return prices_df, self.price_updates

    def retrieve_contracts(self, ticker, currency, security_type):

        def watcher(msg):
            print msg

        def contract_details_handler(msg):
            contracts.append(msg.contractDetails.m_summary)

        def retrieve_data_end_handler(msg):
            self.data_wait = False

        self.tws_conn.registerAll(watcher)
        self.tws_conn.register(contract_details_handler, 'ContractDetails')
        self.tws_conn.register(retrieve_data_end_handler, 'ContractDetailsEnd')

        # self.tws_conn.connect()

        contract = Contract()
        contract.m_exchange = "SMART"
        contract.m_secType = security_type
        contract.m_symbol = ticker
        # contract.m_multiplier = "100"
        contract.m_currency = currency

        self.tws_conn.reqContractDetails(1, contract)

        contracts = []

        self.data_wait = True
        i = 0
        while self.data_wait and i < 90:
            print(i)
            i += 1
            time.sleep(1)

        contracts_df = IBApi.process_contracts(contracts=contracts,
                                               security_type=security_type)

        return contracts_df, contracts

    @staticmethod
    def process_contracts(contracts=None, security_type=None):

        contracts_df = None

        if security_type == 'OPT':

            cols = ['ticker', 'underlying', 'strike', 'maturity_date',
                    'option_type']
            contracts_df = pd.DataFrame(index=range(0, len(contracts)),
                                        columns=cols)
            for i in range(0, len(contracts)):
                contracts_df.loc[i] = [contracts[i].m_localSymbol.replace(" ", ""),
                                       contracts[i].m_symbol,
                                       contracts[i].m_strike,
                                       contracts[i].m_expiry,
                                       contracts[i].m_right]

            contracts_df['option_type'] = contracts_df['option_type'].str.replace(
                'P', 'put')
            contracts_df['option_type'] = contracts_df['option_type'].str.replace(
                'C', 'call')
            contracts_df['maturity_date'] = pd.to_datetime(
                contracts_df['maturity_date'])

        return contracts_df


'''
--------------------------------------------------------------------------------
EXTERNAL DATA API
--------------------------------------------------------------------------------
'''


class ExternalDataApi(object):

    @staticmethod
    def retrieve_data(data_category=None,
                      start_date=dt.datetime.today(),
                      end_date=dt.datetime.today(),
                      options_dict=None):

        # Any external data API should implement a generic retrieve_data
        # method that suppors all relevant data retrieval
        raise NotImplementedError


'''
--------------------------------------------------------------------------------
FIGI API
--------------------------------------------------------------------------------
'''


class FigiApi(object):
    """
    www.openfigi.com
    Supported identifiers:
    ID_ISIN
    ID_BB_UNIQUE
    ID_SEDOL
    ID_COMMON
    ID_WERTPAPIER
    ID_CUSIP
    ID_BB
    ID_ITALY
    ID_EXCH_SYMBOL
    ID_FULL_EXCHANGE_SYMBOL
    COMPOSITE_ID_BB_GLOBAL
    ID_BB_GLOBAL_SHARE_CLASS_LEVEL
    ID_BB_SEC_NUM_DES
    ID_BB_GLOBAL
    TICKER
    ID_CUSIP_8_CHR
    OCC_SYMBOL
    UNIQUE_ID_FUT_OPT
    OPRA_SYMBOL
    TRADING_SYSTEM_IDENTIFIER
    """

    base_url = 'https://api.openfigi.com/v1/mapping'
    api_key = os.getenv('FIGI_API_KEY')

    @classmethod
    def retrieve_mapping(cls, id_type=None, ids=None, exch_codes=None):
        request_dict = list()
        for i in range(0, len(ids)):
            request_dict.append({"idType": id_type,
                                 "idValue": ids[i],
                                 "exchCode": exch_codes[i]})
        return cls.retrieve_mapping_by_dict(request_dict=request_dict)

    @classmethod
    def retrieve_mapping_by_dict(cls, request_dict=None):
        r = requests.post(cls.base_url,
                          headers={"Content-Type": "text/json",
                                   "X-OPENFIGI-APIKEY": cls.api_key},
                          json=request_dict)
        mapping = r.json()
        data = None
        if len(mapping) > 0:
            for i in range(0, len(mapping)):
                if 'data' in mapping[i].keys():
                    if data is None:
                        data = pd.DataFrame.from_dict(mapping[i]['data'],
                                                      orient='columns')
                    else:
                        data = data.append(pd.DataFrame.from_dict(
                            mapping[i]['data'],
                            orient='columns'))

            column_map = {'compositeFIGI': 'composite_figi_id',
                          'exchCode': 'exchange_code',
                          'marketSector': 'bbg_sector',
                          'figi': 'figi_id',
                          'securityType': 'security_sub_type'}
            data = data.rename(columns=column_map)

        return data


class BarchartApi(ExternalDataApi):

    # Set API key
    barchart_api_key = os.getenv('BARCHART_API_KEY')
    base_url = "http://marketdata.websol.barchart.com/"

    @staticmethod
    def retrieve_data(data_category=None,
                      start_date=dt.datetime.today(),
                      end_date=dt.datetime.today(),
                      options_dict=None):

        # Any external data API should implement a generic retrieve_data
        # method that suppors all relevant data retrieval
        raise NotImplementedError

    @classmethod
    def retrieve_quote_data(cls, tickers=None):

        tickers_string = ",".join(tickers)

        barchart_api = "getQuote"
        return_format = "json"
        req_url = cls.base_url + barchart_api + "." + return_format + "?key=" \
                + cls.barchart_api_key + "&symbols=" + tickers_string

        response = urllib2.urlopen(req_url)
        results_dict = json.loads(response.read())
        df = None
        if 'results' in results_dict:

            results_list = results_dict['results']
            df = pd.DataFrame(results_list)

            col_map = {'close': 'close_price',
                       'lastPrice': 'last_price',
                       'high': 'high_price',
                       'open': 'open_price',
                       'low': 'low_price',
                       'netChange': 'change',
                       'tradeTimestamp': 'date'}
            df = df.rename(columns=col_map)

            fields = ['name', 'close_price', 'high_price', 'low_price',
                      'last_price', 'low_price', 'open_price', 'volume']

            df.index = [df['symbol'], df['date']]
            df = df[fields]

        return df


'''
--------------------------------------------------------------------------------
QUANDL API
--------------------------------------------------------------------------------
'''


class QuandlApi(ExternalDataApi):

    # Set quandl API
    ql.ApiConfig.api_key = os.getenv('QUANDL_API_KEY')

    @staticmethod
    def retrieve_data(data_category=None,
                      start_date=dt.datetime.today(),
                      end_date=dt.datetime.today(),
                      options_dict=None):

        # Any external data API should implement a generic retrieve_data
        # method that suppors all relevant data retrieval
        raise NotImplementedError

    @staticmethod
    def get_equity_index_universe():
        return ['DOW', 'SPX', 'NDX_C', 'NDX', 'UKX']

    @staticmethod
    def get_futures_universe():
        base_url = 'https://s3.amazonaws.com/quandl-static-content/'
        quandl_futures = base_url + 'Ticker+CSV%27s/Futures/meta.csv'
        tickers = QuandlApi.retrieve_universe(path=quandl_futures,
                                              filename='futures.csv')
        return tickers

    @staticmethod
    def get_currency_universe():
        base_url = 'https://s3.amazonaws.com/quandl-static-content/'
        quandl_currencies = base_url + 'Ticker+CSV%27s/currencies.csv'
        tickers = QuandlApi.retrieve_universe(
            path=quandl_currencies,
            filename='currencies.csv'
        )
        return tickers

    @staticmethod
    def get_equity_universe(index_ticker):

        base_url = 'https://s3.amazonaws.com/static.quandl.com/tickers/'

        quandl_universe = {'INDU': base_url + 'dowjonesA.csv',
                           'SPX': base_url + 'SP500.csv',
                           'NDX_C': base_url + 'NASDAQComposite.csv',
                           'NDX': base_url + 'nasdaq100.csv',
                           'UKX': base_url + 'FTSE100.csv'}

        quandl_filenames = {'INDU': 'dowjonesA.csv',
                            'SPX': 'SP500.csv',
                            'NDX_C': 'NASDAQComposite.csv',
                            'NDX': 'nasdaq100.csv',
                            'UKX': 'FTSE100.csv'}

        tickers = QuandlApi.retrieve_universe(
            path=quandl_universe[index_ticker],
            filename=quandl_filenames[index_ticker])

        if index_ticker == 'UKX':
            tickers = [ticker + '.LN' for ticker in tickers]

        return tickers

    @staticmethod
    def get_equity_index_identifiers():

        indices = {'SPX': 'YAHOO/INDEX_GSPC',
                   'NDX': 'NASDAQOMX/COMP',
                   'INDU': 'BCB/UDJIAD1',
                   'RTY': 'YAHOO/INDEX_RUI',
                   'SHCOMP': 'YAHOO/INDEX_SSEC',
                   'HSI': 'YAHOO/INDEX_HSI',
                   'NKY': 'NIKKEI/INDEX',
                   'DAX': 'YAHOO/INDEX_GDAXI',
                   'CAC': 'YAHOO/INDEX_FCHI',
                   'IBOV': 'BCB/7',
                   'SPTSX': 'YAHOO/INDEX_GSPTSE',
                   'IBEX': 'YAHOO/INDEX_IBEX',
                   'KOSPI2': 'YAHOO/INDEX_KS11'}

        return indices

    @staticmethod
    def get(quandl_codes=None, start_date=None, end_date=None):

        raw_data = ql.get(quandl_codes,
                          start_date=start_date,
                          end_date=end_date)
        return raw_data

    @staticmethod
    def get_data(tickers=None, start_date=None, end_date=dt.datetime.today()):

        """
        The idea here is to automatically figure out what fields Quandl is
        returning and get back a data structure that's sensible for that
        :param tickers:
        :param start_date:
        :param end_date:
        :return:
        """

        # Quandl API call
        raw_data = ql.get(tickers,
                          start_date=start_date,
                          end_date=end_date)

        # Column names are an amalgam of ticker and field
        data = raw_data.stack()
        col_names = data.index.get_level_values(1).tolist()
        long_tickers, fields = zip(*[s.split(' - ') for s in col_names])

        data.index.names = ['date', 'ticker']
        data = data.reset_index()
        data['ticker'] = long_tickers
        data['field'] = fields
        data.index = [data['ticker'], data['date'], data['field']]
        del data['ticker']
        del data['date']
        del data['field']
        data = data.unstack('field')

        return data

    @staticmethod
    def map_futures_columns(df=None):
        rename_cols = {'Close': 'close_price',
                       'High': 'high_price',
                       'Low': 'low_price',
                       'Open': 'open_price',
                       'Settle': 'settle_price',
                       'Prev. Day Open Interest': 'open_interest',
                       'Total Volume': 'volume'}
        df = df.rename(columns=rename_cols)
        return df

    @staticmethod
    def retrieve_historical_generic_futures_prices(dataset=None,
                                                   futures_series=None,
                                                   source_series=None,
                                                   contract_range=None,
                                                   start_date=None):

        # Obnoxious, need to manage this stuff somewhere
        date_field = 'Date'
        if dataset in ('CHRIS', 'CBOE'):
            date_field = 'Trade Date'
        if dataset == 'CHRIS' and futures_series == 'FVS':
            date_field = 'Date'

        generic_ticker_range = contract_range
        generic_tickers_short = [futures_series + str(i)
                                 for i in generic_ticker_range]
        generic_tickers = [dataset + '/' + source_series + str(i)
                           for i in generic_ticker_range]
        futures_data = pd.DataFrame()

        i = 0
        for ticker in generic_tickers:
            try:
                tmp = ql.get(ticker, start_date=start_date)
            except:
                continue

            tmp['ticker'] = generic_tickers_short[i]
            tmp = tmp[np.isfinite(tmp['Settle'])]
            tmp = tmp.reset_index()
            tmp.index = [tmp['ticker'], tmp[date_field]]
            tmp['contract_number'] = contract_range[i]

            if len(futures_data) == 0:
                futures_data = tmp
            else:
                futures_data = futures_data.append(
                    tmp)

            i += 1

        if len(futures_data) > 0:
            futures_data = futures_data[
                futures_data[date_field] >= start_date]
            del futures_data['ticker']
            del futures_data[date_field]

        else:
            logging.info("no data retrieved...")

        return futures_data

    @staticmethod
    def retrieve_historical_vix_futures_prices(
            start_date=dt.datetime(2007, 3, 24)):

        dataset = 'CBOE'
        futures_series = 'VX'
        futures_data = QuandlApi.retrieve_historical_futures_prices(
            start_date=start_date,
            dataset=dataset,
            futures_series=futures_series
        )
        return futures_data

    @staticmethod
    def retrieve_historical_vstoxx_futures_prices(
            start_date=dt.datetime(2012, 1, 1)):

        dataset = 'EUREX'
        futures_series = 'FVS'
        futures_data = QuandlApi.retrieve_historical_futures_prices(
            start_date=start_date,
            dataset=dataset,
            futures_series=futures_series
        )
        return futures_data

    @staticmethod
    def update_daily_futures_prices(date=None,
                                    dataset=None,
                                    futures_series=None,
                                    contract_range=np.arange(1, 10)):

        if date is None:
            date = utils.workday(dt.datetime.today(), -1)

        # Figure out which months to request
        month = date.month
        year = date.year
        months = list()
        years = list()
        for c in contract_range:

            months.append(month)
            years.append(year)

            month += 1
            if month > 12:
                month = 1
                year += 1

        futures_tickers = list()
        futures_tickers_df = pd.DataFrame(columns=['year', 'month'])

        for i in range(0, len(months)):
            year = years[i]
            month = months[i]

            short_ticker = futures_series + constants.futures_month_codes[
                month] + str(year)
            ticker = dataset + "/" + futures_series \
                     + constants.futures_month_codes[month] + str(year)
            futures_tickers.append(ticker)
            futures_tickers_df.loc[ticker, 'year'] = year
            futures_tickers_df.loc[ticker, 'month'] = month
            futures_tickers_df.loc[ticker, 'short_ticker'] = short_ticker

        start_date = date - BDay(1)
        futures_data = QuandlApi.get_data(futures_tickers,
                                          start_date=start_date)

        if 0 in futures_data.columns:
            futures_data = futures_data[0]

        futures_data = QuandlApi.map_futures_columns(futures_data)

        futures_tickers_df.index.names = ['ticker']
        futures_data = futures_tickers_df[['short_ticker']] \
            .join(futures_data) \
            .reset_index()

        futures_data.index = [futures_data['short_ticker'], futures_data['date']]
        futures_data.index.names = ['ticker', 'date']
        del futures_data['ticker']
        del futures_data['short_ticker']
        del futures_data['date']

        return futures_data

    @staticmethod
    def retrieve_historical_futures_prices(start_date=None,
                                           dataset=None,
                                           futures_series=None):

        date_field = 'Date'
        if dataset == 'CBOE':
            date_field = 'Trade Date'

        years = np.arange(start_date.year, dt.datetime.today().year + 2)

        futures_data = pd.DataFrame()
        futures_tickers = list()
        futures_tickers_df = pd.DataFrame(columns=['year', 'month'])
        for year in years:
            for month in constants.futures_month_codes:

                short_ticker = futures_series + constants.futures_month_codes[
                    month] + str(year)
                ticker = dataset + "/" + futures_series \
                         + constants.futures_month_codes[month] + str(year)
                futures_tickers.append(ticker)
                futures_tickers_df.loc[ticker, 'year'] = year
                futures_tickers_df.loc[ticker, 'month'] = month

                try:
                    tmp = ql.get(ticker)
                except:
                    continue

                tmp['ticker'] = short_ticker
                tmp = tmp[np.isfinite(tmp['Settle'])]
                tmp = tmp.reset_index()
                tmp.index = [tmp['ticker'], tmp[date_field]]

                if len(futures_data) == 0:
                    futures_data = tmp
                else:
                    futures_data = futures_data.append(tmp)
        futures_data = futures_data[
            futures_data[date_field] >= start_date]
        del futures_data['ticker']
        del futures_data[date_field]

        futures_data = QuandlApi.map_futures_columns(futures_data)
        futures_data.index.names = ['ticker', 'date']

        return futures_data

    @staticmethod
    def retrieve_universe(path, filename):
        opener = urllib.URLopener()
        target = path
        opener.retrieve(target, filename)
        tickers_file = pd.read_csv(filename)
        if 'ticker' in tickers_file:
            tickers = tickers_file['ticker'].values
        else:
            tickers=tickers_file
        return tickers

'''
--------------------------------------------------------------------------------
YAHOO API
--------------------------------------------------------------------------------
'''


class YahooApi(ExternalDataApi):

    @staticmethod
    def map_price_columns(prices_df):

        column_map = {'Open': 'open_price',
                      'High': 'high_price',
                      'Low': 'low_price',
                      'Close': 'last_price',
                      'Volume': 'volume',
                      'Adj Close': 'adj_close'}

        prices_df = prices_df.rename(columns=column_map)
        return prices_df

    @staticmethod
    def retrieve_data(data_category=None,
                      start_date=dt.datetime.today(),
                      end_date=dt.datetime.today(),
                      options_dict=None):

        raise NotImplementedError

    @staticmethod
    def prepare_date_strings(date):

        date_yr = date.year.__str__()
        date_mth = date.month.__str__()
        date_day = date.day.__str__()

        if len(date_mth) == 1: date_mth = '0' + date_mth
        if len(date_day) == 1: date_day = '0' + date_day

        return date_yr, date_mth, date_day

    @staticmethod
    def retrieve_prices(equity_tickers=None,
                        start_date=None,
                        end_date=dt.datetime.today(),
                        change_to_data_frame_from_panel=True):

        data = pdata.get_data_yahoo(symbols=equity_tickers,
                                    start=start_date,
                                    end=end_date)

        if change_to_data_frame_from_panel:
            if isinstance(data, pd.Panel):
                data = data.to_frame()

        data = YahooApi.map_price_columns(data)

        return data

    @staticmethod
    def retrieve_dividends(equity_tickers=None,
                           start_date=None,
                           end_date=dt.datetime.today()):
        output = pd.DataFrame(columns=['Date', 'Ticker', 'Dividend'])
        for ticker in equity_tickers:
            dataset = YahooApi.retrieve_dividend(equity_ticker=ticker,
                                                 start_date=start_date,
                                                 end_date=end_date)
            if dataset is None:
                continue
            else:
                dataset['Ticker'] = ticker
                output = output.append(dataset)
        # output.index = [output['Ticker'], output['Date']]
        # del output['Ticker']
        # del output['Date']
        return output

    @staticmethod
    def retrieve_dividend(equity_ticker=None,
                          start_date=None,
                          end_date=dt.datetime.today()):

        start_year, start_month, start_day = \
            YahooApi.prepare_date_strings(start_date)
        end_year, end_month, end_day = \
            YahooApi.prepare_date_strings(end_date)

        url = 'http://finance.yahoo.com/q/hp?s='
        url = url + equity_ticker
        url = url + '&a=' + start_month
        url = url + '&b=' + start_day
        url = url + '&c=' + start_year
        url = url + '&d=' + end_month
        url = url + '&e=' + end_day
        url = url + '&f=' + end_year
        url = url + '&g=v'

        html = urllib.urlopen(url=url).read()
        soup = BeautifulSoup(html)
        table = soup.find("table", attrs={"class": "yfnc_datamodoutline1"})

        if table is None:
            return None

        headings = [th.get_text() for th in table.find("tr").find_all("th")]

        datasets = []
        rows = table.find_all("tr")[1:]

        for row in rows:
            try:
                dataset = zip(headings, (td.get_text()
                                         for td in row.find_all("td")))
                date = dataset[0][1]
                raw_dividend = dataset[1][1]
                dividend = float(raw_dividend.__str__()
                                 .replace(' Dividend', ''))
                datasets.append([date, dividend])
            except:
                x = 1
        dividends = pd.DataFrame(data=datasets, columns=['Date', 'Dividend'])
        dividends['Date'] = dividends['Date'].apply(dateutil.parser.parse)
        # dividends.index = dividends['Date']

        return dividends

    @staticmethod
    def option_ticker_from_attrs(df):
        df['Expiry'] = df.index.get_level_values(level='Expiry')
        df['Strike'] = df.index.get_level_values(level='Strike')
        df['Type'] = df.index.get_level_values(level='Type')
        df['Type'] = df['Type'].str[0].str.upper()
        ticker = df['Underlying'] + " " \
               + df['Expiry'].map(dt.datetime.date).map(str) \
               + " " + df['Type'] + df['Strike'].map(str)
        return ticker

    @staticmethod
    def retrieve_options_data(equity_ticker, above_below=20):

        expiry_dates, links = pdata.YahooOptions(equity_ticker) \
            ._get_expiry_dates_and_links()
        expiry_dates = [date for date in expiry_dates
                        if date >= dt.datetime.today().date()]

        options_data = pd.DataFrame()
        for date in expiry_dates:

            print('retrieving options data for ' + equity_ticker
                  + ' on ' + date.__str__())

            data = pdata.YahooOptions(equity_ticker).get_near_stock_price(
                above_below=above_below,
                call=True,
                put=True,
                expiry=date)

            # Irritating: method seems to return incorrect expiration dates
            data = data.reset_index()
            data['Expiry'] = date
            data.index = [data['Strike'], data['Expiry'],
                          data['Type'], data['Symbol']]

            options_data = options_data.append(data)

        return options_data, expiry_dates

    @staticmethod
    def process_options_data(raw_options_data=None, expiry_dates=None):
        x=1

    @staticmethod
    def extract_attributes_from_option_symbol(symbol, underlying_ticker):
        symbol = symbol.replace(underlying_ticker, '')
        year = 2000 + int(symbol[0:2])
        month = int(symbol[2:4])
        day = int(symbol[4:6])
        maturity_date = dt.datetime(year, month, day)
        option_type = symbol[6]
        if option_type == 'C':
            option_type = 'call'
        elif option_type == 'P':
            option_type = 'put'
        strike = float(symbol[7:len(symbol)+1]) / 1000
        return option_type, maturity_date, strike

    @staticmethod
    def transform_options_data(options_data, expiry_dates, risk_free_rate):

        # Calculate forwards

        return 1

    @staticmethod
    def get_equity_index_identifiers():

        universe_mapping = {
            'SPX': '^GSPC',
            'RTY': '^RUT',
            'INDU': '^DJI',
            'CAC': '^FCHI',
            'DAX': '^GDAXI',
            'UKX': '^FTSE',
            'NDX': '^NDX',
            'NDX_C': '^IXIC',
            'HSI': '^HSI',
            'HSCEI': '^HSCE',
            'NKY': '^N225',
            'SPTSX': '^GSPTSE',
            'AS51': '^AXJO',
            'SHCOMP': '^SSE',
            'KOSPI2': '^KS200'
        }

        return universe_mapping


'''
--------------------------------------------------------------------------------
DATA SCRAPER
--------------------------------------------------------------------------------
'''


class DataScraper(object):

    @staticmethod
    def update_vix_settle_price(overwrite_date=None):

        # temp filename to use
        filename = "data/vix_settle.csv"

        # URL for daily settle prices
        url = "http://cfe.cboe.com/data/DailyVXFuturesEODValues/DownloadFS.aspx"
        urllib.urlretrieve(url, filename)
        data = pd.read_csv(filename + ".csv")

        data = data.reset_index()

        series, dates = zip(*[mystr.split(" ")
                              for mystr in data['Symbol'].tolist()])
        dates = pd.to_datetime(dates)

        # Only include monthly futures, ignore weeklies
        included_indices = list()
        prices = pd.DataFrame(index=included_indices,
                              columns=['date', 'maturity_date', 'settle_price'])

        date = utils.workday(dt.datetime.today(), -1)
        if overwrite_date is not None:
            date = overwrite_date

        for i in range(0, len(dates)):
            if series[i] == 'VX':
                included_indices.append(i)
                prices.loc[i, 'date'] = date
                prices.loc[i, 'maturity_date'] = dates[i].date()
                prices.loc[i, 'settle_price'] = data.loc[i, 'SettlementPrice']

        return prices
