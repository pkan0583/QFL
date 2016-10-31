import pandas as pd
import numpy as np
import simplejson as json
import datetime as dt

import qfl.utilities.basic_utilities as utils
from qfl.core.database_interface import DatabaseInterface as db
from qfl.core.database_interface import DatabaseUtilities as dbutils
from qfl.core.market_data import VolatilitySurfaceManager
from qfl.strategies.strategy_master import StrategyMaster
import qfl.core.market_data as md
import logging

db.initialize()
logger = logging.getLogger()


def initialize_strategy_environment(start_date=md.history_default_start_date,
                                    **kwargs):

    # Volatility
    vsm = VolatilitySurfaceManager()
    tickers = md.get_etf_vol_universe()

    load_clean_data = kwargs.get('load_clean_data', False)
    if load_clean_data:
        clean_data = pd.read_excel('data/clean_ivol_data.xlsx')
        clean_data['date'] = clean_data['date'].ffill()
        clean_data = clean_data.set_index(['date', 'ticker'], drop=True)
        vsm.data = clean_data
    else:
        vsm.load_data(tickers=tickers, start_date=start_date)
        clean_data, final_tickers = vsm.clean_data(tickers=tickers)
        vsm.data = clean_data

    # Strategy master
    sm = StrategyMaster()

    # VIX strategies
    sm.initialize_vix_curve(price_field='settle_price')
    sm.initialize_equity_vs_vol()

    # Volatility RV
    sm.initialize_macro_model()
    sm.initialize_volatility_factor_model()
    sm.initialize_volatility_rv(vsm=vsm)

    return sm


def get_model_id(model_name=None):

    df = db.get_data(table_name='models')
    df = df[df['model_name'] == model_name]
    if len(df) == 0:
        raise ValueError('model name not found!')
    else:
        return int(df['id'].values[0])


def get_model_param_config_id(model_name=None, settings=None):

    export_settings = model_settings_to_json(settings=settings)
    model_params = get_model_param_configs(model_name=model_name)

    if export_settings not in model_params['model_params'].tolist():
        archive_model_param_config(model_name=model_name, settings=settings)
        model_params = get_model_param_configs(model_name=model_name)
    row = model_params[model_params['model_params'] == export_settings]
    id = row['id'].values[0]
    return id


def get_model_param_configs(model_name=None):
    model_id = get_model_id(model_name)
    df = db.get_data(table_name='model_param_config',
                     where_str='model_id ={0}'.format(model_id))
    df['model_params'] = df['model_params'].astype(str)
    return df


def get_model_output_fields(model_name=None):
    model_id = get_model_id(model_name)
    df = db.get_data(table_name='model_output_config',
                     where_str='model_id = {0}'.format(model_id))
    return df


def get_models():

    s = 'select * from models'
    df = db.read_sql(s)

    return df


def archive_models():

    df = pd.DataFrame(columns=['id', 'model_name', 'ref_object_type'])
    df.loc[0] = [1, 'vix_curve', 'strategy']
    df.loc[1] = [2, 'equity_vs_vol', 'strategy']
    df.loc[2] = [3, 'vol_rv', 'equity']

    existing_data = get_models()
    ind = df.index[~df['model_name'].isin(existing_data['model_name'].tolist())]
    df = df.loc[ind]

    table = db.get_table('models')
    db.execute_bulk_insert(df=df, table=table)


def archive_model_output_config():

    df = pd.DataFrame(columns=['model_id', 'output_name'])

    # VIX Curve
    model_id = get_model_id('vix_curve')
    df.loc[0] = [model_id, 'pnl_net']
    df.loc[1] = [model_id, 'pnl_gross']
    df.loc[2] = [model_id, 'pnl_static']
    df.loc[3] = [model_id, 'position']

    # Equity vs vol
    model_id = get_model_id('equity_vs_vol')
    df.loc[4] = [model_id, 'pnl_net']
    df.loc[5] = [model_id, 'pnl_gross']
    df.loc[6] = [model_id, 'pnl_static']
    df.loc[7] = [model_id, 'position']

    # Volatility RV
    model_id = get_model_id('vol_rv')
    df.loc[8] = [model_id, 'pnl_net']
    df.loc[9] = [model_id, 'pnl_gross']
    df.loc[10] = [model_id, 'vega_net']
    df.loc[11] = [model_id, 'vega_gross']

    df['id'] = df.index.get_level_values(None)

    table = db.get_table('model_output_config')
    db.execute_db_save(df=df,
                       table=table)


def model_settings_to_json(settings=None):

    export_settings = settings.copy()
    for setting in export_settings:
        if utils.is_date_type(export_settings[setting]):
            export_settings[setting] = export_settings[setting].date().__str__()
        elif isinstance(export_settings[setting], (np.ndarray, np.generic)):
            export_settings[setting] = export_settings[setting].tolist()
    export_settings = json.dumps(export_settings)
    return export_settings


def archive_model_param_config(model_name=None, settings=None):

    export_settings = model_settings_to_json(settings)

    model_id = get_model_id(model_name)
    model_params = get_model_param_configs(model_name)

    if export_settings not in model_params['model_params'].tolist():
        new_id = db.read_sql('select max(id) from model_param_config')
        try:
            new_id = new_id['max'][0] + 1
        except:
            new_id = 0
        model_params = pd.DataFrame(columns=['id', 'model_id', 'model_params'])
        model_params.loc[0] = [new_id, model_id, export_settings]
        table = db.get_table('model_param_config')
        db.execute_bulk_insert(df=model_params, table=table)


def archive_model_outputs(model=None, outputs=None):

    # Get id for param config
    param_config_id = get_model_param_config_id(
        model_name=model.name,
        settings=model.settings)

    # Ready to archive
    df = model.process_model_outputs(outputs)
    df['model_param_id'] = param_config_id
    df['model_output_id'] = 0
    df['model_id'] = get_model_id(model.name)

    # Output fields for this model
    output_fields = get_model_output_fields(model.name)
    output_dict = dict()
    for i in range(0, len(output_fields)):
        field = str(output_fields.iloc[i]['output_name'])
        output_dict[field] = output_fields.iloc[i]['id']
        ind = df.index[df['output_name'] == field]
        df.loc[ind, 'model_output_id'] = output_dict[field]

    table = db.get_table('model_outputs')
    db.execute_db_save(df=df, table=table, time_series=True)


def get_model_outputs(model_name=None,
                      settings=None,
                      start_date=md.history_default_start_date,
                      end_date=dt.datetime.today()):

    param_config_id = get_model_param_config_id(
        model_name=model_name, settings=settings)

    s = "select * from model_outputs_view "
    s += " where model_name = '{0}'".format(model_name)
    s += ' and model_param_id = {0}'.format(param_config_id)

    data = md._get_time_series_data(s=s,
                                    start_date=start_date,
                                    end_date=end_date)

    return data


def get_strategy_signals(model_name=None):

    model_id = get_model_id(model_name)
    s = db.get_data(table_name='model_signals',
                    where_str='model_id = {0}'.format(model_id))
    return s


def archive_strategy_signals(model=None, signal_data=None):

    signal_names = signal_data.columns
    model_id = get_model_id(model.name)
    existing_signals = get_strategy_signals(model_name=model.name)

    df = pd.DataFrame(columns=['model_id', 'signal_name'])
    df['signal_name'] = signal_names
    df['model_id'] = model_id

    # Filter down now
    ind = df.index[
        ~df['signal_name'].isin(existing_signals['signal_name'].tolist())]
    df = df.loc[ind]

    if len(df) > 0:
        table = db.get_table('model_signals')
        db.execute_bulk_insert(df=df, table=table)


def archive_strategy_signal_data(model=None,
                                 signal_data=None,
                                 signal_data_z=None):

    signals = get_strategy_signals(model_name=model.name)
    model_param_id = get_model_param_config_id(model_name=model.name,
                                               settings=model.settings)

    df = model.process_signal_data_for_archive(signal_data=signal_data,
                                               signal_data_z=signal_data_z,
                                               db_signals=signals,
                                               model_param_id=model_param_id)
    table = db.get_table('model_signal_data')
    db.execute_db_save(df=df, table=table, time_series=True)


def archive_strategy_signal_pnl(model=None,
                                signal_pnl=None):

    signals = get_strategy_signals(model_name=model.name)
    model_param_id = get_model_param_config_id(model_name=model.name,
                                               settings=model.settings)

    df = model.process_signal_pnl_for_archive(signal_pnl=signal_pnl,
                                              db_signals=signals,
                                              model_param_id=model_param_id)
    table = db.get_table('model_signal_data')
    db.execute_db_save(df=df, table=table, time_series=True)


def archive_portfolio_strategy_signal_data(model=None,
                                           signal_data=None,
                                           signal_data_z=None,
                                           backtest_update_start_date=None):

    if backtest_update_start_date is not None:
        dates = signal_data.index.get_level_values('date')
        signal_data = signal_data[dates >= backtest_update_start_date]
        dates = signal_data_z.index.get_level_values('date')
        signal_data_z = signal_data_z[dates >= backtest_update_start_date]

    if len(signal_data) > 0:

        signals = get_strategy_signals(model_name=model.name)
        model_param_id = get_model_param_config_id(model_name=model.name,
                                                   settings=model.settings)

        df = model.process_signal_data_for_archive(
            signal_data=signal_data,
            signal_data_z=signal_data_z,
            db_signals=signals,
            model_param_id=model_param_id)

        table = db.get_table('model_signal_data')
        db.execute_db_save(df=df, table=table, time_series=True)


def get_portfolio_strategy_signal_pnl(model=None,
                                      signal_names=None,
                                      start_date=md.history_default_start_date,
                                      end_date=dt.datetime.today()):

    df = get_strategy_signal_data(model=model,
                                  signal_names=signal_names,
                                  ref_entity_ids=['strategy'],
                                  start_date=start_date,
                                  end_date=end_date)
    df =  df[['date', 'signal_name', 'pnl']]\
        .set_index(['date', 'signal_name'], drop=True)\
        .unstack('signal_name')\
        ['pnl']

    return df


def get_strategy_signal_data(model=None,
                             ref_entity_ids=None,
                             signal_names=None,
                             start_date=md.history_default_start_date,
                             end_date=dt.datetime.today()):

    param_config_id = get_model_param_config_id(
        model_name=model.name, settings=model.settings)

    if signal_names is None:
        s = "select * from model_signal_data_view "
    else:
        s = "select {0} from model_signal_data_view ".format(\
            dbutils.format_for_query(signal_names))
    s += " where model_name = '{0}'".format(model.name)
    s += ' and model_param_id = {0}'.format(param_config_id)

    if ref_entity_ids is not None:
        s += ' and ref_entity_id in {0}'.format(
            dbutils.format_for_query(ref_entity_ids))

    data = md._get_time_series_data(s=s,
                                    start_date=start_date,
                                    end_date=end_date)

    return data


def compute_and_archive_portfolio_strategy_signal_pnl(model=None,
                                                      signal_data=None,
                                                      **kwargs):

    backtest_update_start_date = kwargs.get('backtest_update_start_date', None)

    archive_model_param_config(model_name=model.name, settings=model.settings)
    archive_strategy_signals(model=model, signal_data=signal_data)

    signal_names = kwargs.get('signal_names', None)
    if signal_names is None:
        signal_names = signal_data.columns

    for signal in signal_names:

        print('now running ' + signal + ' ...')

        signal_data_s = pd.DataFrame(signal_data[signal])

        portfolio_summary_df = model.compute_signal_backtests(
            signal_data=signal_data_s,
            signal_name=signal,
            backtest_update_start_date=backtest_update_start_date)

        portfolio_summary_df.index.names = ['signal_name', 'date']

        signal_pnl = pd.DataFrame(portfolio_summary_df['pnl_gross']) \
            .rename(columns={'pnl_gross': 'pnl'}).unstack('signal_name')

        archive_strategy_signal_pnl(model=model, signal_pnl=signal_pnl)


def archive_portfolio_strategy_security_master(model=None, sec_master=None):

    # Attributes json
    attribute_fields = ['underlying', 'start_date', 'maturity_date', 'strike']

    sec_master = sec_master.copy(deep=True)
    sec_master['start_date'] = sec_master['start_date'].astype(str)
    sec_master['maturity_date'] = sec_master['maturity_date'].astype(str)

    sec_master_archive = utils.json_column_from_columns(
        df=sec_master,
        columns=attribute_fields,
        new_col_name='instrument_attributes')

    sec_master_archive['model_id'] = get_model_id(model_name=model.name)
    sec_master_archive['model_param_id'] = get_model_param_config_id(
        model_name=model.name, settings=model.settings)
    sec_master_archive['instrument_type'] = 'volatility_swap'
    sec_master_archive = sec_master_archive.rename(
        columns={'instrument': 'instrument_name'})

    sec_master_archive = sec_master_archive[['model_id',
                                             'model_param_id',
                                             'instrument_name',
                                             'instrument_type',
                                             'instrument_attributes']]

    table = db.get_table('model_portfolio_sec_master')
    db.execute_db_save(df=sec_master_archive,
                       table=table,
                       use_column_as_key='instrument_name')


def get_portfolio_strategy_security_master(model=None):

    model_id = get_model_id(model.name)
    model_param_id = get_model_param_config_id(model_name=model.name,
                                               settings=model.settings)
    s_m = db.read_sql("select * from model_portfolio_sec_master "
                      + " where model_id = {0} and model_param_id = {1}"
                      .format(model_id, model_param_id))

    return s_m


def archive_portfolio_strategy_positions(model=None, positions=None):

    model_id = get_model_id(model_name=model.name)
    model_param_id = get_model_param_config_id(model_name=model.name,
                                               settings=model.settings)
    db_sec_master = get_portfolio_strategy_security_master(model=model)

    positions_archive = model.process_positions_for_archive(
        positions=positions,
        db_sec_master=db_sec_master,
        model_id=model_id,
        model_param_id=model_param_id
    )
    table = db.get_table('model_portfolio_outputs')
    db.execute_db_save(df=positions_archive, table=table, time_series=True)


def compute_and_archive_standard_strategy_backtest(sm=None,
                                                   strategy_name=None):

    model = sm.strategies[strategy_name]
    signal_data = sm.outputs[strategy_name]['signal_data']
    signal_data_z = sm.outputs[strategy_name]['signal_data_z']
    signal_pnl = sm.outputs[strategy_name]['signal_pnl']

    # These need to be done
    archive_model_param_config(model_name=model.name,
                               settings=model.settings)
    archive_strategy_signals(model=model,
                             signal_data=signal_data)

    # This is the signals and Z-scores, at the signal/strategy level
    archive_strategy_signal_data(model=model,
                                 signal_data=signal_data,
                                 signal_data_z=signal_data_z)

    # This is the signal PNL, at the signal/strategy level
    archive_strategy_signal_pnl(model=model,
                                signal_pnl=signal_pnl)

    # This is the final PNL, at the strategy level
    archive_model_outputs(model=model,
                          outputs=sm.outputs[strategy_name])

def compute_and_archive_portfolio_strategy_backtest(sm=None,
                                                    strategy_name=None,
                                                    included_signals=None,
                                                    start_date=dt.datetime(1990,1,1)):
    model = sm.strategies[strategy_name]
    signal_data = sm.outputs[strategy_name]['signal_data']
    signal_data_z = sm.outputs[strategy_name]['signal_data_z']

    # These need to be done
    archive_model_param_config(model_name=model.name,
                               settings=model.settings)
    archive_strategy_signals(model=model,
                             signal_data=signal_data)

    # This is the signal PNL, at the signal/strategy level
    compute_and_archive_portfolio_strategy_signal_pnl(
        model=model,
        signal_data=signal_data,
        backtest_update_start_date=start_date)

    # This is the signals and Z-scores, at the signal/underlying level
    archive_portfolio_strategy_signal_data(
        model=model,
        signal_data=signal_data,
        signal_data_z=signal_data_z,
        backtest_update_start_date=start_date)

    # Retrieve signal PNL
    signal_pnl = get_strategy_signal_data(model=model,
                                          ref_entity_ids=['strategy',
                                                             't'])
    signal_pnl = signal_pnl[['date', 'signal_name', 'pnl']] \
        .sort_values(['signal_name', 'date']) \
        .set_index(['date', 'signal_name'])['pnl'] \
        .unstack('signal_name')

    # Run the optimization and the final backtest
    positions, portfolio_summary, sec_master, optim_output = \
        model.compute_master_backtest(
            signal_pnl=signal_pnl,
            signal_data=signal_data,
            included_signals=included_signals,
            backtest_update_start_date=start_date
        )

    # Archive the portfolio summary (strategy-level PNL)
    archive_model_outputs(model=model, outputs=portfolio_summary)

    # Archive the security master
    archive_portfolio_strategy_security_master(model=model,
                                               sec_master=sec_master)

    # Archive the positions
    archive_portfolio_strategy_positions(model=model,
                                         positions=positions)