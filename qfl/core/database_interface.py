import os
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import BDay
import quandl as ql
import numpy as np
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, JSONB
from sqlalchemy.sql.expression import bindparam
import requests
import logging
import simplejson as json


import qfl.core.constants as constants
import qfl.utilities.basic_utilities as utils

"""
-------------------------------------------------------------------------------
DATABASE UTILITIES
-------------------------------------------------------------------------------
"""


class DatabaseUtilities(object):

    @staticmethod
    def format_for_query(item):

        output = None
        if isinstance(item, str):
            item = [item]

        if len(item) == 1:
            if isinstance(item[0], str):
                output = "('" + item[0] + "')"
            elif isinstance(item[0], int):
                output = '(' + str(item[0]) + ')'
            else:
                output = '(' + item[0] + ')'
        else:
            output = tuple(item)

        return output

    @staticmethod
    def parenthetical_string_list_with_quotes(items):
        out = "("
        if not isinstance(items, (tuple, list)):
            items = [items]
        counter = 0
        for item in items:
            out += "'" + item + "'"
            counter += 1
            if counter == len(items):
                out += ')'
            else:
                out += ','
        return out

    @staticmethod
    def build_pk_where_or_str(df, table, use_column_as_key):
        where_str = str()
        if use_column_as_key is None:
            key_column_names = table.primary_key.columns.keys()
        else:
            key_column_names = [use_column_as_key]
        for i in range(0, len(df)):
            if where_str != '':
                where_str += ') or ('
            else:
                where_str += '('
            for key in key_column_names:
                if key in df:
                    key_to_write = str(df.iloc[i][key])
                    if isinstance(table.columns[key].type, sa.String):
                        key_to_write = "'" + key_to_write + "'"
                    if isinstance(table.columns[key].type, sa.Date):
                        key_to_write = "'" + str(pd.to_datetime(key_to_write)) + "'"
                    if key != key_column_names[0]:
                        where_str += ' and '
                    where_str += key + " = " + key_to_write
        if where_str != '':
            where_str += ')'

        return where_str, key_column_names

    @staticmethod
    def build_pk_where_str(df=None,
                           table=None,
                           use_column_as_key=None,
                           time_series=False,
                           date_column_name='date'):

        # This is the guy we're building
        where_str = str()

        # Get key column names
        if use_column_as_key is None:
            key_column_names = table.primary_key.columns.keys()
        else:
            key_column_names = [use_column_as_key]

        for key in key_column_names:
            if key in df:

                if time_series and key == date_column_name:

                    dates = pd.to_datetime(np.unique(df[date_column_name]))
                    min_date = dates.min().__str__()
                    max_date = dates.max().__str__()

                    # Chaining the and clauses
                    if where_str != '':
                        where_str += ' and '

                    where_str += " date >= '" + min_date + "'"
                    where_str += " and date <= '" + max_date + "'"

                else:

                    # Only include the unique values of this key
                    unique_keys = np.unique(df[key])
                    if len(unique_keys) == 1:
                        t = '(' + str(unique_keys[0]) + ')'
                    else:
                        t = tuple(unique_keys)

                    # Handle unicode etc.
                    if isinstance(table.columns[key].type, sa.String):
                        unique_keys = [str(k) for k in unique_keys]
                        t = DatabaseUtilities.parenthetical_string_list_with_quotes(
                            unique_keys)

                    # Handle dates
                    if isinstance(table.columns[key].type, sa.Date):
                        unique_keys = [str(k) for k in pd.to_datetime(unique_keys)]
                        t = DatabaseUtilities.parenthetical_string_list_with_quotes(
                            unique_keys)

                    # Chaining the and clauses
                    if where_str != '':
                        where_str += ' and '

                    # Making sure unique_keys is iterable
                    if len(unique_keys) == 1:
                        try:
                            a = iter(unique_keys)
                            unique_keys = unique_keys[0]
                        except:
                            a = 1

                    # Add the finalized format
                    where_str += key + " in {0}".format(t)

        return where_str, key_column_names

    @staticmethod
    def get_data(_db=None,
                 table_name=None,
                 index_table=False,
                 parse_dates=False,
                 columns=None,
                 where_str=None):
        if table_name in _db.tables:

            get_whole_table = False
            if (columns is None) and (where_str is None):
                get_whole_table = True

            table = _db.tables[table_name]

            pk_columns = None
            if index_table:
                pk_columns = table.primary_key.columns.keys()

            if columns is None:
                columns = table.columns.keys()

            parse_dates_columns = None

            if parse_dates:
                for column in columns:
                    if table.columns[column].type == sa.sql.sqltypes.Date:
                        parse_dates_columns.append(table.columns[column].name)

            sql = "select " + ", ".join(columns) + " from " + table_name
            if where_str is not None:
                where_str = where_str.replace('where', '')
                sql += " where " + where_str

            if get_whole_table:

                output = pd.read_sql(sql=table_name,
                                     con=_db.engine,
                                     index_col=pk_columns,
                                     parse_dates=parse_dates_columns,
                                     columns=columns)

            else:

                output = pd.read_sql(sql=sql,
                                     con=_db.engine,
                                     index_col=pk_columns,
                                     parse_dates=parse_dates_columns)

            return output

"""
-------------------------------------------------------------------------------
DATABASE INTERFACE
-------------------------------------------------------------------------------
"""


class DatabaseInterface(object):

    # SqlAlchemy metadata
    connection_string = None
    engine = None
    conn = None
    metadata = None
    tables = None

    @classmethod
    def read_sql(cls, query, parse_dates=None):
        output = pd.read_sql(sql=query,
                             con=cls.engine,
                             parse_dates=parse_dates)
        return output

    @classmethod
    def initialize(cls):

        cls.connection_string = os.getenv('POSTGRES_CONN_STRING')
        cls.engine = sa.create_engine(cls.connection_string, echo=False)

        # Connect to the database
        cls.conn = cls.engine.connect()

        # Metadata object
        cls.metadata = sa.MetaData()

        # Create tables
        cls.define_qfl_tables()

    @classmethod
    def define_qfl_tables(cls):

        # Tables
        cls.tables = dict()

        # TIME SERIES
        time_series_table = sa.Table(
            'time_series', cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('field', sa.String(128), primary_key=True),
            sa.Column('value', sa.Float, primary_key=False))

        cls.tables['time_series'] = time_series_table

        # SECURITIES
        securities_table = sa.Table(
            'securities', cls.metadata,
            sa.Column('figi_id', sa.String(64), primary_key=True),
            sa.Column('composite_figi_id', sa.String(64), nullable=False),
            sa.Column('ticker', sa.String(64), nullable=False),
            sa.Column('security_type', sa.String(16), nullable=False),
            sa.Column('security_sub_type', sa.String(16), primary_key=False),
            sa.Column('exchange_code', sa.String(8), primary_key=False),
            sa.Column('bbg_sector', sa.String(16), nullable=False),
            sa.Column('name', sa.String(128), nullable=False),
            sa.Column('active', sa.Boolean))
        cls.tables['securities'] = securities_table

        # DATA SOURCES
        data_sources_table = sa.Table(
            'data_sources', cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('name', sa.String(128), nullable=False))
        cls.tables['data_sources'] = data_sources_table

        # SECURITY IDENTIFIERS
        security_identifiers_table = sa.Table(
            'security_identifiers', cls.metadata,
            sa.Column('data_source_id', sa.Integer, primary_key=True),
            sa.Column('figi_id', sa.String(64), primary_key=True),
            sa.Column('identifier_type', sa.String(64), primary_key=True),
            sa.Column('identifier_value', sa.String(64), nullable=False),
            sa.ForeignKeyConstraint(['figi_id'], ['securities.figi_id']),
            sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id']))
        cls.tables['security_identifiers'] = security_identifiers_table

        # INDEX IDENTIFIERS
        index_identifiers_table = sa.Table(
            'index_identifiers', cls.metadata,
            sa.Column('data_source_id', sa.Integer, primary_key=True),
            sa.Column('index_id', sa.Integer, primary_key=True),
            sa.Column('identifier_type', sa.String(64), primary_key=True),
            sa.Column('identifier_value', sa.String(64), nullable=False),
            sa.ForeignKeyConstraint(['index_id'], ['equity_indices.index_id']))
        cls.tables['index_identifiers'] = index_identifiers_table

        # EQUITY INDEX PRICES
        equity_index_prices_table = sa.Table(
            'equity_index_prices', cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('adj_close', sa.Float, primary_key=False),
            sa.Column('last_price', sa.Float, primary_key=False),
            sa.Column('open_price', sa.Float, primary_key=False),
            sa.Column('high_price', sa.Float, primary_key=False),
            sa.Column('low_price', sa.Float, primary_key=False),
            sa.Column('volume', sa.Float, primary_key=False),
            sa.ForeignKeyConstraint(['id'], ['equity_indices.index_id']))
        cls.tables['equity_index_prices'] = equity_index_prices_table

        # GENERIC INDICES
        generic_indices_table = sa.Table(
            'generic_indices', cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('ticker', sa.String(16), unique=True),
            sa.Column('name', sa.String(128), unique=True),
            sa.Column('currency', sa.String(16))
        )
        cls.tables['generic_indices'] = generic_indices_table

        # GENERIC INDEX PRICES
        generic_index_prices_table = sa.Table(
            'generic_index_prices', cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('adj_close', sa.Float, primary_key=False),
            sa.Column('last_price', sa.Float, primary_key=False),
            sa.Column('open_price', sa.Float, primary_key=False),
            sa.Column('high_price', sa.Float, primary_key=False),
            sa.Column('low_price', sa.Float, primary_key=False),
            sa.Column('volume', sa.Float, primary_key=False),
            sa.ForeignKeyConstraint(['id'], ['generic_indices.id']))
        cls.tables['generic_index_prices'] = generic_index_prices_table

        # EQUITIES
        equities_table = sa.Table(
            'equities', cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True, unique=True),
            sa.Column('ticker', sa.String(64), nullable=False),
            sa.Column('exchange_code', sa.String(8), unique=False),
            sa.UniqueConstraint('ticker', 'exchange_code', name='equities_uix_1')
        )
        cls.tables['equities'] = equities_table

        # EQUITY OPTIONS
        equity_options_table = sa.Table(
            'equity_options', cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('ticker', sa.String(64), nullable=False, unique=True),
            sa.Column('underlying_id', sa.Integer, primary_key=False),
            sa.Column('option_type', sa.String(16), primary_key=False),
            sa.Column('strike_price', sa.Float, primary_key=False),
            sa.Column('maturity_date', sa.Date, primary_key=False),
            sa.ForeignKeyConstraint(['underlying_id'], ['equities.id']))
        cls.tables['equity_options'] = equity_options_table

        # EQUITY PRICES
        equity_prices_table = sa.Table(
            'equity_prices', cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('adj_close', sa.Float, primary_key=False),
            sa.Column('last_price', sa.Float, primary_key=False),
            sa.Column('bid_price', sa.Float, primary_key=False),
            sa.Column('ask_price', sa.Float, primary_key=False),
            sa.Column('open_price', sa.Float, primary_key=False),
            sa.Column('high_price', sa.Float, primary_key=False),
            sa.Column('low_price', sa.Float, primary_key=False),
            sa.Column('volume', sa.Float, primary_key=False),
            sa.ForeignKeyConstraint(['id'], ['equities.id']))
        cls.tables['equity_prices'] = equity_prices_table

        # EQUITY OPTION PRICES
        equity_option_prices_table = sa.Table(
            'equity_option_prices',
            cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('last_price', sa.Float, primary_key=False),
            sa.Column('bid_price', sa.Float, primary_key=False),
            sa.Column('ask_price', sa.Float, primary_key=False),
            sa.Column('iv', sa.Float, primary_key=False),
            sa.Column('volume', sa.Float, primary_key=False),
            sa.Column('open_interest', sa.Float, primary_key=False),
            sa.Column('spot_price', sa.Float, primary_key=False),
            sa.Column('quote_time', sa.Time, primary_key=False),
            sa.ForeignKeyConstraint(['id'], ['equity_options.id']))
        cls.tables['equity_option_prices'] = equity_option_prices_table

        # EQUITY SCHEDULES
        equity_schedules_table = sa.Table(
            'equity_schedules',
            cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('schedule_type', sa.String(16), primary_key=True),
            sa.Column('value', sa.Float, primary_key=False),
            sa.ForeignKeyConstraint(['id'], ['equities.id']))
        cls.tables['equity_schedules'] = equity_schedules_table

        # EQUITY INDICES
        equity_index_table = sa.Table(
            'equity_indices',
            cls.metadata,
            sa.Column('index_id', sa.Integer, primary_key=True, unique=True),
            sa.Column('ticker', sa.String(32), primary_key=False),
            sa.Column('country', sa.String(8), primary_key=False))
        cls.tables['equity_indices'] = equity_index_table

        # EQUITY INDEX MEMBERS
        equity_index_members_table = sa.Table(
            'equity_index_members',
            cls.metadata,
            sa.Column('index_id', sa.Integer, primary_key=True),
            sa.Column('equity_id', sa.Integer, primary_key=True),
            sa.Column('valid_date', sa.Date, primary_key=True),
            sa.ForeignKeyConstraint(['index_id'], ['equity_indices.index_id']),
            sa.ForeignKeyConstraint(['equity_id'], ['equities.id']))
        cls.tables['equity_index_members'] = equity_index_members_table

        # FUTURES SERIES
        futures_series_table = sa.Table(
            'futures_series',
            cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('series', sa.String(128), nullable=False),
            sa.Column('description', sa.String(128), primary_key=False),
            sa.Column('exchange', sa.String(128), primary_key=False),
            sa.Column('currency', sa.String(16), nullable=False),
            sa.Column('contract_size', sa.String(128), primary_key=False),
            sa.Column('units', sa.String(128), primary_key=False),
            sa.Column('point_value', sa.Float, primary_key=False),
            sa.Column('tick_value', sa.Float, primary_key=False),
            sa.Column('trading_times', sa.String),
            sa.Column('start_date', sa.Date),
            sa.Column('delivery_months', sa.String(16), primary_key=False),
            sa.Column('cftc_code', sa.String(16), primary_key=False)
        )
        cls.tables['futures_series'] = futures_series_table

        # FUTURES CONTRACTS
        futures_contracts_table = sa.Table(
            'futures_contracts',
            cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('series_id', sa.Integer, nullable=False),
            sa.Column('ticker', sa.String(32), nullable=False),
            sa.Column('maturity_date', sa.Date, nullable=False),
            sa.ForeignKeyConstraint(['series_id'], ['futures_series.id'])
        )
        cls.tables['futures_contracts'] = futures_contracts_table

        # FUTURES PRICES
        futures_prices_table = sa.Table(
            'futures_prices',
            cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('last_price', sa.Float, primary_key=False),
            sa.Column('bid_price', sa.Float, primary_key=False),
            sa.Column('ask_price', sa.Float, primary_key=False),
            sa.Column('settle_price', sa.Float, primary_key=False),
            sa.Column('open_price', sa.Float, primary_key=False),
            sa.Column('close_price', sa.Float, primary_key=False),
            sa.Column('high_price', sa.Float, primary_key=False),
            sa.Column('low_price', sa.Float, primary_key=False),
            sa.Column('seasonality_adj_price', sa.Float, primary_key=False),
            sa.Column('price_change', sa.Float),
            sa.Column('open_interest', sa.Integer, primary_key=False),
            sa.Column('volume', sa.Integer, primary_key=False),
            sa.Column('days_to_maturity', sa.Integer, primary_key=False),
            sa.ForeignKeyConstraint(['id'], ['futures_contracts.id'])
        )
        cls.tables['futures_prices'] = futures_prices_table

        # GENERIC FUTURES CONTRACTS
        generic_futures_contracts_table = sa.Table(
            'generic_futures_contracts',
            cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('series_id', sa.Integer, nullable=False),
            sa.Column('ticker', sa.String(32), nullable=False),
            sa.Column('contract_number', sa.Integer, nullable=False),
            sa.ForeignKeyConstraint(['series_id'], ['futures_series.id'])
        )
        cls.tables['generic_futures_contracts'] = generic_futures_contracts_table

        # GENERIC FUTURES PRICES
        generic_futures_prices_table = sa.Table(
            'generic_futures_prices',
            cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('last_price', sa.Float, primary_key=False),
            sa.Column('bid_price', sa.Float, primary_key=False),
            sa.Column('ask_price', sa.Float, primary_key=False),
            sa.Column('settle_price', sa.Float, primary_key=False),
            sa.Column('open_price', sa.Float, primary_key=False),
            sa.Column('close_price', sa.Float, primary_key=False),
            sa.Column('high_price', sa.Float, primary_key=False),
            sa.Column('low_price', sa.Float, primary_key=False),
            sa.Column('seasonality_adj_price', sa.Float, primary_key=False),
            sa.Column('open_interest', sa.Integer, primary_key=False),
            sa.Column('volume', sa.Integer, primary_key=False),
            sa.Column('futures_contract_id', sa.Integer, primary_key=False),
            sa.Column('days_to_maturity', sa.Integer, primary_key=False),
            sa.ForeignKeyConstraint(['id'], ['generic_futures_contracts.id'])
        )
        cls.tables['generic_futures_prices'] = generic_futures_prices_table

        # CONSTANT MATURITY FUTURES PRICES
        constant_maturity_futures_prices_table = sa.Table(
            'constant_maturity_futures_prices', cls.metadata,
            sa.Column('series_id', sa.Integer, primary_key=True),
            sa.Column('days_to_maturity', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('price', sa.Float),
            sa.ForeignKeyConstraint(['series_id'], ['futures_series.id'])
        )
        cls.tables['constant_maturity_futures_prices'] \
            = constant_maturity_futures_prices_table

        # STAGING TABLE FOR ORATS
        orats_staging_table = sa.Table(
            'staging_orats', cls.metadata,
            sa.Column('ticker', sa.String(32), primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('underlying_price', sa.Float),
            sa.Column('iv_1m', sa.Float),
            sa.Column('iv_2m', sa.Float),
            sa.Column('iv_3m', sa.Float),
            sa.Column('iv_1mc', sa.Float),
            sa.Column('days_to_maturity_1mc', sa.Float),
            sa.Column('iv_2mc', sa.Float),
            sa.Column('days_to_maturity_2mc', sa.Float),
            sa.Column('iv_3mc', sa.Float),
            sa.Column('days_to_maturity_3mc', sa.Float),
            sa.Column('iv_4mc', sa.Float),
            sa.Column('days_to_maturity_4mc', sa.Float),
            sa.Column('skew', sa.Float),
            sa.Column('curvature', sa.Float),
            sa.Column('skew_inf', sa.Float),
            sa.Column('curvature_inf', sa.Float),
            sa.Column('rv_10d', sa.Float),
            sa.Column('rv_20d', sa.Float),
            sa.Column('rv_60d', sa.Float),
            sa.Column('rv_120d', sa.Float),
            sa.Column('rv_252d', sa.Float),
            sa.Column('tick_rv_10d', sa.Float),
            sa.Column('tick_rv_20d', sa.Float),
            sa.Column('tick_rv_60d', sa.Float),
            sa.Column('tick_rv_120d', sa.Float),
            sa.Column('tick_rv_252d', sa.Float)
        )
        cls.tables['staging_orats'] = orats_staging_table

        # CLEAN EQUITY IMPLIED VOLATILITY
        equity_implied_volatility_table = sa.Table(
            'equity_implied_volatility', cls.metadata,
            sa.Column('ticker', sa.String(32), primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('underlying_price', sa.Float),
            sa.Column('iv_1m', sa.Float),
            sa.Column('iv_2m', sa.Float),
            sa.Column('iv_3m', sa.Float),
            sa.Column('iv_1mc', sa.Float),
            sa.Column('days_to_maturity_1mc', sa.Float),
            sa.Column('iv_2mc', sa.Float),
            sa.Column('days_to_maturity_2mc', sa.Float),
            sa.Column('iv_3mc', sa.Float),
            sa.Column('days_to_maturity_3mc', sa.Float),
            sa.Column('iv_4mc', sa.Float),
            sa.Column('days_to_maturity_4mc', sa.Float),
            sa.Column('skew', sa.Float),
            sa.Column('curvature', sa.Float),
            sa.Column('skew_inf', sa.Float),
            sa.Column('curvature_inf', sa.Float),
            sa.Column('rv_10d', sa.Float),
            sa.Column('rv_20d', sa.Float),
            sa.Column('rv_60d', sa.Float),
            sa.Column('rv_120d', sa.Float),
            sa.Column('rv_252d', sa.Float),
            sa.Column('tick_rv_10d', sa.Float),
            sa.Column('tick_rv_20d', sa.Float),
            sa.Column('tick_rv_60d', sa.Float),
            sa.Column('tick_rv_120d', sa.Float),
            sa.Column('tick_rv_252d', sa.Float)
        )
        cls.tables['equity_implied_volatility'] \
            = equity_implied_volatility_table

        # STAGING TABLES FOR OPTIONWORKS
        optionworks_ivm_staging_table = sa.Table(
            'staging_optionworks_ivm', cls.metadata,
            sa.Column('code', sa.String(64), primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('future', sa.Float),
            sa.Column('atm', sa.Float),
            sa.Column('rr25', sa.Float),
            sa.Column('rr10', sa.Float),
            sa.Column('fly25', sa.Float),
            sa.Column('fly10', sa.Float),
            sa.Column('beta1', sa.Float),
            sa.Column('beta2', sa.Float),
            sa.Column('beta3', sa.Float),
            sa.Column('beta4', sa.Float),
            sa.Column('beta5', sa.Float),
            sa.Column('beta6', sa.Float),
            sa.Column('minmoney', sa.Float),
            sa.Column('maxmoney', sa.Float),
            sa.Column('dte', sa.Float),
            sa.Column('dtt', sa.Float))
        cls.tables['staging_optionworks_ivm'] = optionworks_ivm_staging_table

        optionworks_ivs_staging_table = sa.Table(
            'staging_optionworks_ivs', cls.metadata,
            sa.Column('code', sa.String(64), primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('dnsvol', sa.Float),
            sa.Column('p01dvol', sa.Float),
            sa.Column('p05dvol', sa.Float),
            sa.Column('p10dvol', sa.Float),
            sa.Column('p15dvol', sa.Float),
            sa.Column('p20dvol', sa.Float),
            sa.Column('p25dvol', sa.Float),
            sa.Column('p30dvol', sa.Float),
            sa.Column('p35dvol', sa.Float),
            sa.Column('p40dvol', sa.Float),
            sa.Column('p45dvol', sa.Float),
            sa.Column('p50dvol', sa.Float),
            sa.Column('p55dvol', sa.Float),
            sa.Column('p60dvol', sa.Float),
            sa.Column('p65dvol', sa.Float),
            sa.Column('p70dvol', sa.Float),
            sa.Column('p75dvol', sa.Float),
            sa.Column('p80dvol', sa.Float),
            sa.Column('p85dvol', sa.Float),
            sa.Column('p90dvol', sa.Float),
            sa.Column('p95dvol', sa.Float),
            sa.Column('p99dvol', sa.Float),
            sa.Column('c01dvol', sa.Float),
            sa.Column('c05dvol', sa.Float),
            sa.Column('c10dvol', sa.Float),
            sa.Column('c15dvol', sa.Float),
            sa.Column('c20dvol', sa.Float),
            sa.Column('c25dvol', sa.Float),
            sa.Column('c30dvol', sa.Float),
            sa.Column('c35dvol', sa.Float),
            sa.Column('c40dvol', sa.Float),
            sa.Column('c45dvol', sa.Float),
            sa.Column('c50dvol', sa.Float),
            sa.Column('c55dvol', sa.Float),
            sa.Column('c60dvol', sa.Float),
            sa.Column('c65dvol', sa.Float),
            sa.Column('c70dvol', sa.Float),
            sa.Column('c75dvol', sa.Float),
            sa.Column('c80dvol', sa.Float),
            sa.Column('c85dvol', sa.Float),
            sa.Column('c90dvol', sa.Float),
            sa.Column('c95dvol', sa.Float),
            sa.Column('c99dvol', sa.Float))
        cls.tables['staging_optionworks_ivs'] = optionworks_ivs_staging_table

        # OPTIONWORKS

        futures_series_identifiers_table = sa.Table(
            'futures_series_identifiers', cls.metadata,
            sa.Column('source', sa.String(64), primary_key=True),
            sa.Column('source_id', sa.String(64), primary_key=True),
            sa.Column('series_id', sa.Integer, nullable=False)
        )
        cls.tables['futures_series_identifiers'] \
            = futures_series_identifiers_table

        optionworks_codes_table = sa.Table(
            'optionworks_codes', cls.metadata,
            sa.Column('ow_code', sa.String(64), primary_key=True),
            sa.Column('ow_data_type', sa.String(8), nullable=False),
            sa.Column('maturity_type', sa.String(32), nullable=False),
            sa.Column('exchange_code', sa.String(32), nullable=False),
            sa.Column('futures_series', sa.String(32), nullable=False),
            sa.Column('option', sa.String(32), nullable=False),
            sa.Column('futures_contract', sa.String(32)),
            sa.Column('days_to_maturity', sa.Integer))
        cls.tables['optionworks_codes'] = optionworks_codes_table

        futures_ivol_fixed_maturity_by_delta_table = sa.Table(
            'futures_ivol_fixed_maturity_by_delta', cls.metadata,
            sa.Column('series_id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('futures_contract', sa.String(32), primary_key=True),
            sa.Column('option_type', sa.String(16), primary_key=True),
            sa.Column('ivol_01d', sa.Float),
            sa.Column('ivol_05d', sa.Float),
            sa.Column('ivol_10d', sa.Float),
            sa.Column('ivol_15d', sa.Float),
            sa.Column('ivol_20d', sa.Float),
            sa.Column('ivol_25d', sa.Float),
            sa.Column('ivol_30d', sa.Float),
            sa.Column('ivol_35d', sa.Float),
            sa.Column('ivol_40d', sa.Float),
            sa.Column('ivol_45d', sa.Float),
            sa.Column('ivol_50d', sa.Float),
            sa.Column('ivol_55d', sa.Float),
            sa.Column('ivol_60d', sa.Float),
            sa.Column('ivol_65d', sa.Float),
            sa.Column('ivol_70d', sa.Float),
            sa.Column('ivol_75d', sa.Float),
            sa.Column('ivol_80d', sa.Float),
            sa.Column('ivol_85d', sa.Float),
            sa.Column('ivol_90d', sa.Float),
            sa.Column('ivol_95d', sa.Float),
            sa.Column('ivol_99d', sa.Float),
        )
        cls.tables['futures_ivol_fixed_maturity_by_delta'] \
            = futures_ivol_fixed_maturity_by_delta_table

        futures_ivol_constant_maturity_by_delta_table = sa.Table(
            'futures_ivol_constant_maturity_by_delta', cls.metadata,
            sa.Column('series_id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('days_to_maturity', sa.Integer, primary_key=True),
            sa.Column('option_type', sa.String(16), primary_key=True),
            sa.Column('ivol_01d', sa.Float),
            sa.Column('ivol_05d', sa.Float),
            sa.Column('ivol_10d', sa.Float),
            sa.Column('ivol_15d', sa.Float),
            sa.Column('ivol_20d', sa.Float),
            sa.Column('ivol_25d', sa.Float),
            sa.Column('ivol_30d', sa.Float),
            sa.Column('ivol_35d', sa.Float),
            sa.Column('ivol_40d', sa.Float),
            sa.Column('ivol_45d', sa.Float),
            sa.Column('ivol_50d', sa.Float),
            sa.Column('ivol_55d', sa.Float),
            sa.Column('ivol_60d', sa.Float),
            sa.Column('ivol_65d', sa.Float),
            sa.Column('ivol_70d', sa.Float),
            sa.Column('ivol_75d', sa.Float),
            sa.Column('ivol_80d', sa.Float),
            sa.Column('ivol_85d', sa.Float),
            sa.Column('ivol_90d', sa.Float),
            sa.Column('ivol_95d', sa.Float),
            sa.Column('ivol_99d', sa.Float)
        )
        cls.tables['futures_ivol_constant_maturity_by_delta'] \
            = futures_ivol_constant_maturity_by_delta_table

        futures_ivol_fixed_maturity_surface_model_table = sa.Table(
            'futures_ivol_fixed_maturity_surface_model', cls.metadata,
            sa.Column('series_id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('futures_contract', sa.String(32), primary_key=True),
            sa.Column('future', sa.Float),
            sa.Column('atm', sa.Float),
            sa.Column('rr25', sa.Float),
            sa.Column('rr10', sa.Float),
            sa.Column('fly25', sa.Float),
            sa.Column('fly10', sa.Float),
            sa.Column('beta1', sa.Float),
            sa.Column('beta2', sa.Float),
            sa.Column('beta3', sa.Float),
            sa.Column('beta4', sa.Float),
            sa.Column('beta5', sa.Float),
            sa.Column('beta6', sa.Float),
            sa.Column('minmoney', sa.Float),
            sa.Column('maxmoney', sa.Float),
            sa.Column('dte', sa.Float),
            sa.Column('dtt', sa.Float))
        cls.tables['futures_ivol_fixed_maturity_surface_model'] \
            = futures_ivol_fixed_maturity_surface_model_table

        futures_ivol_constant_maturity_surface_model_table = sa.Table(
            'futures_ivol_constant_maturity_surface_model', cls.metadata,
            sa.Column('series_id', sa.Integer, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('days_to_maturity', sa.Integer, primary_key=True),
            sa.Column('future', sa.Float),
            sa.Column('atm', sa.Float),
            sa.Column('rr25', sa.Float),
            sa.Column('rr10', sa.Float),
            sa.Column('fly25', sa.Float),
            sa.Column('fly10', sa.Float),
            sa.Column('beta1', sa.Float),
            sa.Column('beta2', sa.Float),
            sa.Column('beta3', sa.Float),
            sa.Column('beta4', sa.Float),
            sa.Column('beta5', sa.Float),
            sa.Column('beta6', sa.Float),
            sa.Column('minmoney', sa.Float),
            sa.Column('maxmoney', sa.Float),
            sa.Column('dte', sa.Float),
            sa.Column('dtt', sa.Float))
        cls.tables['futures_ivol_constant_maturity_surface_model'] \
            = futures_ivol_constant_maturity_surface_model_table

        # EXCHANGES
        exchanges_table = sa.Table(
            'exchanges', cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('exchange_code', sa.String(16), unique=True, nullable=False),
            sa.Column('name', sa.String(128), nullable=False),
            sa.Column('country', sa.String(16), nullable=False),
        )
        cls.tables['exchanges'] = exchanges_table

        # COUNTRIES
        countries_table = sa.Table(
            'countries', cls.metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('country_code', sa.String(16), unique=True, nullable=False),
            sa.Column('name', sa.String(128), nullable=False)
        )
        cls.tables['countries'] = countries_table

        # YAHOO CODES
        yfinance_metadata_table = sa.Table(
            'yfinance_metadata', cls.metadata,
            sa.Column('yahoo_code', sa.String(64), primary_key=True),
            sa.Column('name', sa.String(128))
        )
        cls.tables['yfinance_metadata'] = yfinance_metadata_table

        # MODEL OUTPUTS
        model_outputs_table = sa.Table(
            'model_outputs', cls.metadata,
            sa.Column('model', sa.String(64), primary_key=True),
            sa.Column('output_id', sa.String(64), primary_key=True),
            sa.Column('model_config', JSONB, primary_key=True),
            sa.Column('date', sa.Date, primary_key=True),
            sa.Column('value', sa.Float)
        )
        cls.tables['model_outputs'] = model_outputs_table

    @classmethod
    def create_tables(cls):
        cls.metadata.create_all(cls.engine)

    @classmethod
    def get_table(cls, table_name=None):
        if cls.tables is not None:
            if table_name in cls.tables:
                return cls.tables.get(table_name)
            else:
                return None

    @classmethod
    def execute_bulk_insert(cls, df=None, table=None, batch_size=1000):

        # Very first step: replace NaN with null
        df = df.where((pd.notnull(df)), None)

        num_batches = int(np.ceil(float(len(df)) / batch_size))
        result = None
        for i in range(0, num_batches):
            urange = np.min([(i + 1) * batch_size, len(df)-1])
            if urange == 0:
                list_to_write = df.to_dict(orient='records')
            else:
                ind = np.arange(i * batch_size, urange)
                list_to_write = df.iloc[ind].to_dict(orient='records')
            result = cls.engine.execute(sa.insert(table=table,
                                                  values=list_to_write))
        return result


    @classmethod
    def execute_db_save(cls,
                        df=None,
                        table=None,
                        use_column_as_key=None,
                        extra_careful=False,
                        time_series=False):

        """
        This is a generic fast method to insert-if-exists-update a database
        table using SqlAlchemy expression language framework.
        :param df: DataFrame containing data to be written to table
        :param table: SqlAlchemy table object
        :param use_column_as_key string
        :param extra_careful bool
        :param delete_existing bool
        :param time_series bool
        :return: none
        """

        logging.info('starting archive of '
                     + str(len(df)) + ' records')

        logging.info(df.head(5))

        # Very first step: replace NaN with null
        df = df.where((pd.notnull(df)), None)

        # Now blow up index
        try:
            df = df.reset_index()
        except:
            df = df.reset_index(drop=True)

        # Get where string to identify existing rows
        if extra_careful:
            where_str, key_column_names = DatabaseUtilities.build_pk_where_or_str(
                df=df, table=table, use_column_as_key=use_column_as_key)
        else:
            where_str, key_column_names = DatabaseUtilities.build_pk_where_str(
                df=df,
                table=table,
                use_column_as_key=use_column_as_key,
                time_series=time_series)

        # Grab column names
        column_names = table.columns.keys()
        column_list = [table.columns[column_name]
                       for column_name in column_names]

        # Grab the existing data in table corresponding to the new data
        s = sa.select(columns=column_list).where(where_str)
        existing_data = pd.read_sql(sql=s,
                                    con=cls.engine,
                                    index_col=key_column_names)

        # Add index to df so that we can identify existing data
        key_column_names_in_df = list()
        for key_column in key_column_names:
            if key_column in df.columns:
                key_column_names_in_df.append(key_column)
        df_key_cols = [df[key_column] for key_column in key_column_names_in_df]

        if use_column_as_key is None:
            if len(key_column_names_in_df) > 0:
                df.index = df_key_cols
        else:
            orig_key_cols = table.primary_key.columns.keys()
            df.index = df[use_column_as_key]
            existing_data = existing_data.reset_index()
            existing_data.index = existing_data[use_column_as_key]

        # Now we figure out what part of the df is already in existing_data
        insert_df = df[~df.index.isin(existing_data.index)]
        update_df = df[df.index.isin(existing_data.index)]

        # Clean up extra columns
        for key_col in key_column_names:
            if key_col in insert_df.columns:
                del insert_df[key_col]

        # Insert part is easy
        if len(insert_df) > 0:
            insert_df = insert_df.reset_index()
            cls.execute_bulk_insert(insert_df, table)

        # Generic version of update using bindparams
        if len(update_df) > 0:

            update_df_mod = update_df

            # If we used an alternative column for the key (e.g. if the table's
            # primary key is an autoincrement integer) we need to join to get it
            if use_column_as_key is not None:
                join_cols = [use_column_as_key] + orig_key_cols
                try:
                    existing_data = existing_data.reset_index()
                except:
                    existing_data.index = existing_data[orig_key_cols]
                update_df_mod = pd.merge(left=existing_data[join_cols],
                                         right=update_df,
                                         on=key_column_names)

            # Cannot use database column names as bindparams (odd)
            for col in key_column_names:
                update_df_mod = update_df_mod.rename(columns={col: col + "_"})

            # Dictionary format for records
            list_to_write = update_df_mod.to_dict(orient='records')

            # Build the update command
            s = table.update()
            for col in key_column_names:
                s = s.where(table.c[col] == bindparam(col + "_"))
            values_dict = dict()
            for col in key_column_names:
                values_dict[col] = bindparam(col)
            s.values(values_dict)

            # Execute the update command
            cls.conn.execute(s, list_to_write)

            logging.info('completed archive of '
                        + str(len(list_to_write)) + ' records')

    @classmethod
    def get_data(cls,
                 table_name=None,
                 index_table=False,
                 parse_dates=False,
                 columns=None,
                 where_str=None):

        return DatabaseUtilities.get_data(
            _db=cls,
            table_name=table_name,
            index_table=index_table,
            parse_dates=parse_dates,
            columns=columns,
            where_str=where_str
        )

    @classmethod
    def get_index_data_source_identifiers(cls,
                                          data_source_name=None,
                                          tickers=None):

        s = "select * from equity_index_identifiers_view " \
            " where data_source_name = '" + data_source_name + "'"

        if tickers is not None:
            tickers = [str(ticker) for ticker in tickers]
            t = DatabaseUtilities.format_for_query(tickers)
            s += " and ticker in {0}".format(t)

        data = cls.read_sql(s)
        return data

    @classmethod
    def get_equity_indices(cls, tickers=None):

        if tickers is None:
            indices = cls.get_data(table_name='equity_indices')
        else:
            where_str = "ticker in {0}".format(tuple(tickers))
            if len(tickers) == 1:
                where_str = "ticker = '" + tickers[0] + "'"
            indices = cls.get_data(table_name='equity_indices',
                                   where_str=where_str)
        return indices

    @classmethod
    def get_equity_ids(cls, equity_tickers=None):

        ids = list()
        equities = cls.get_data(table_name='equities')

        if equity_tickers is None:

            ids = equities['id'].tolist()

        else:

            equities.index = equities['ticker']

            for ticker in equity_tickers:
                if ticker in equities.index:
                    ids.append(equities.loc[ticker, 'id'])

        return ids

    @classmethod
    def get_equity_tickers(cls, ids=None):

        tickers = list()
        equities = cls.get_data(table_name='equities')

        if ids is None:

            tickers = equities['ticker'].tolist()

        else:

            equities.index = equities['id']

            for id in ids:
                if id in equities.index:
                    tickers.append(equities.loc[id, 'ticker'])

        return tickers

    @classmethod
    def get_etfs(cls):

        etfs = cls.read_sql("select * from equities where ticker in "
                            "(select ticker from securities "
                            "where security_sub_type = 'ETP')")

        return etfs

    @classmethod
    def get_equities(cls, equity_indices=None):

        s = "select * from equities"

        if equity_indices is not None:

            s = "select * from equities e where id in " \
                "(select equity_id from equity_index_members where index_id in " \
                "(select index_id from equity_indices where ticker in {0}))" \
                .format(tuple(equity_indices))

        equities = cls.read_sql(s)
        return equities

    @classmethod
    def get_futures_series(cls, futures_series=None, exchange_code=None):

        where_str = " series in {0}".format(
            DatabaseUtilities.format_for_query(futures_series))
        if exchange_code is not None:
            where_str += " and exchange in {0}".format(
                DatabaseUtilities.format_for_query(exchange_code))
        futures_series_data = cls.get_data(table_name='futures_series',
                                           where_str=where_str)
        return futures_series_data
