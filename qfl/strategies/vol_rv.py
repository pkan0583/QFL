import pandas as pd
import numpy as np
import datetime as dt
from collections import OrderedDict

import statsmodels.api as sm
import qfl.core.market_data as md
import qfl.core.constants as constants
import qfl.utilities.basic_utilities as utils
import qfl.core.calcs as calcs

from qfl.strategies.strategies import Strategy, PortfolioStrategy, PortfolioOptimizer
from qfl.core.market_data import VolatilitySurfaceManager
from qfl.core.marking import MarkingEngine, SecurityMaster


class VolswapRvStrategy(PortfolioStrategy):

    # volatility surface manager
    _vsm = None
    _universe = None
    strat_data = dict()
    calc = dict()
    settings = OrderedDict()
    default_start_date = dt.datetime(2010, 1, 3)

    def __init__(self, **kwargs):

        # Call super
        PortfolioStrategy.__init__(self, 'vol_rv')

        # Default universe
        self.set_universe()

        self.name = 'vol_rv'

        self.set_universe()
        self.initialize_settings(**kwargs)

    def get_default_settings(self):

        default_settings = OrderedDict()
        default_settings['trade_frequency_days'] = 5
        default_settings['start_date'] = self.default_start_date
        default_settings['quantiles'] = [0.10, 0.30, 0.70, 0.90, 1.0]
        default_settings['buy_q'] = 0.90
        default_settings['sell_q'] = 0.30
        default_settings['trade_tenor_month'] = 3
        default_settings['hedge_ratio_vov_weight'] = 0.50
        default_settings['transaction_costs_per_unit'] = 0.25

        return default_settings

    def initialize_settings(self, **kwargs):

        default_settings = self.get_default_settings()

        self.settings['trade_frequency_days'] \
            = kwargs.get('trade_frequency_days',
                         default_settings['trade_frequency_days'])

        self.settings['start_date'] = kwargs.get(
            'start_date', default_settings['start_date'])

        self.settings['backtest_update_start_date'] \
            = kwargs.get('backtest_update_start_date',
                         default_settings['start_date'])

        self.settings['quantiles'] = kwargs.get(
            'quantiles', default_settings['quantiles'])

        self.settings['buy_q'] = kwargs.get('buy_q', default_settings['buy_q'])
        self.settings['sell_q'] = kwargs.get('sell_q', default_settings['sell_q'])

        self.settings['trade_tenor_month'] = kwargs.get(
            'trade_tenor_month', default_settings['trade_tenor_month'])

        self.settings['hedge_ratio_vov_weight'] \
            = kwargs.get('hedge_ratio_vov_weight',
                         default_settings['hedge_ratio_vov_weight'])

        self.settings['transaction_costs_per_unit'] \
            = kwargs.get('transaction_costs_per_unit',
                         default_settings['transaction_costs_per_unit'])

    def set_universe(self, **kwargs):

        self._universe = md.get_etf_vol_universe()

    def initialize_data(self, **kwargs):

        self._vsm = kwargs.get('volatility_surface_manager', None)

        start_date = self.settings['start_date']

        # Set universe
        tickers = self.get_universe(**kwargs)

        # Implied volatility data
        if self._vsm is None:
            raise ValueError('Requires volatility surface manager as input!')
        fields = ['iv_2m',
                  'iv_3m',
                  'days_to_maturity_1mc',
                  'days_to_maturity_2mc',
                  'days_to_maturity_3mc']

        # Stock prices
        stock_price_start_date = utils.workday(
            date=start_date,
            num_days=-constants.trading_days_per_year * 2)
        stock_prices = md.get_equity_prices(tickers=tickers,
                                            start_date=stock_price_start_date)
        stock_prices = stock_prices['adj_close'].unstack(level='ticker')

        # Implied volatility data
        vrv_data = self._vsm.get_data(tickers=tickers,
                                      start_date=start_date,
                                      fields=fields)

        # Add term structure slope
        stock_prices = stock_prices.stack('ticker')
        vrv_data['ts_slope'] = vrv_data['iv_3m'] - vrv_data['iv_2m']
        vrv_data['stock_prices'] = stock_prices

        # Data storage
        self.strat_data['vrv_data'] = vrv_data
        self.strat_data['stock_prices'] = stock_prices
        self.settings['tickers'] = np.unique(
            vrv_data.index.get_level_values('ticker'))

    def get_roll_schedule(self, **kwargs):

        # For now we're going to assume that all names have the same roll sched

        tickers = ['SPY']
        maturity_dates = self._vsm.get_roll_schedule(
            tickers=tickers,
            start_date=self.settings['start_date'])

        return maturity_dates

    def process_data(self, **kwargs):

        vrv_data = self.strat_data['vrv_data']

        # Daily returns
        vrv_data['returns'] = (vrv_data['stock_prices']
                                .unstack('ticker')
                                .diff(1)
                               / vrv_data['stock_prices']
                                .unstack('ticker')
                                .shift(1)) \
                               .stack('ticker')

        # Ivol change
        vrv_data['iv_3m_chg'] = (vrv_data['iv_3m']
                                 .unstack('ticker')
                                 .diff(1)
                                 .stack('ticker'))

        # Get roll schedule
        maturity_dates = self.get_roll_schedule()

        # Three month volswaps
        days_elapsed = 0
        days_to_maturity = 63
        vov_com = kwargs.get('vov_com', 63)

        vrv_data['vol_of_iv_3m'] = (vrv_data['iv_3m_chg']
                                    .unstack('ticker')
                                    .ewm(com=vov_com)
                                    .std()).stack('ticker')

        # Theta (don't really need this, but just for info)
        vrv_data['theta'] = calcs.volswap_theta(iv=vrv_data['iv_3m'],
                                                rv=vrv_data['iv_3m'],
                                                strike=vrv_data['iv_3m'],
                                                days_elapsed=days_elapsed,
                                                total_days=days_to_maturity)

        # Gamma/Theta PNL
        vrv_data['gt_pnl'] = calcs.volswap_daily_pnl(
            daily_return=vrv_data['returns'] * 100.0,
            iv=vrv_data['iv_3m'],
            rv=vrv_data['iv_3m'],
            strike=vrv_data['iv_3m'],
            days_elapsed=days_elapsed,
            total_days=days_to_maturity)

        # Add in IV rolldown PNL and IV change
        vrv_data['tot_pnl'] = vrv_data['gt_pnl'] \
            + vrv_data['iv_3m_chg'] \
            - vrv_data['ts_slope'] / constants.trading_days_per_month

        self.strat_data['vrv_data'] = vrv_data

    def initialize_tick_rv_signals(self, **kwargs):

        diff_in_pct = kwargs.get('diff_in_pct', True)

        tickers = self.get_universe()
        tick_vol = md.get_equity_tick_realized_volatility(
            tickers=tickers,
            start_date=self.settings['start_date'])

        windows = [10, 20, 60, 120, 252]
        cols = [str('tick_diff_') + str(w) for w in windows]
        signal_data = pd.DataFrame(index=self.strat_data['vrv_data'].index,
                                   columns=cols)

        for w in windows:
            sig = 'tick_diff_' + str(w)
            signal_data[sig] = tick_vol['tick_rv_' + str(w) + 'd'] \
                             - tick_vol['rv_' + str(w) + 'd']

            if diff_in_pct:
                signal_data[sig] /= tick_vol['rv_' + str(w) + 'd']

        return signal_data

    def initialize_rv_iv_signals(self, **kwargs):

        rv_com = kwargs.get('rv_com', [10, 21, 42, 63, 126, 252])
        cols = ['rv_iv_' + str(com) for com in rv_com]
        diff_in_pct = kwargs.get('diff_in_pct', True)

        signal_data = pd.DataFrame(index=self.strat_data['vrv_data'].index,
                                   columns=cols)
        self.calc['rv'] = pd.DataFrame(index=self.strat_data['vrv_data'].index,
                                       columns=cols)

        for com in rv_com:
            sig = 'rv_iv_' + str(com)
            self.calc['rv'][com] = (self.strat_data['vrv_data']['returns']
                                        .unstack('ticker')
                                        .ewm(com=com)
                                        .std()
                                        ).stack('ticker') \
                * np.sqrt(constants.trading_days_per_year) * 100.0
            signal_data[sig] = self.calc['rv'][com] \
                               - self.strat_data['vrv_data']['iv_3m']
            if diff_in_pct:
                signal_data[sig] /= self.strat_data['vrv_data']['iv_3m']

        return signal_data

    def initialize_iv_signals(self, **kwargs):

        iv_com = kwargs.get('iv_com', [10, 21, 42, 63, 126, 252])
        cols = ['iv_' + str(com) for com in iv_com]
        diff_in_pct = kwargs.get('diff_in_pct', True)

        signal_data = pd.DataFrame(index=self.strat_data['vrv_data'].index,
                                   columns=cols)

        self.calc['iv_ewm'] = pd.DataFrame(index=self.strat_data['vrv_data'].index,
                                           columns=cols)

        for com in iv_com:
            sig = 'iv_' + str(com)
            self.calc['iv_ewm'][com] = (self.strat_data['vrv_data']['iv_3m']
                                        .unstack('ticker')
                                        .ewm(com=com)
                                        .mean()
                                        ).stack('ticker')
            signal_data[sig] = self.strat_data['vrv_data']['iv_3m'] \
                             - self.calc['iv_ewm'][com]
            if diff_in_pct:
                signal_data[sig] /= self.strat_data['vrv_data']['iv_3m']

        return signal_data

    def initialize_rv_signals(self, **kwargs):

        rv_com = kwargs.get('rv_com', [10, 21, 42, 63, 126, 252])
        cols = ['rv_' + str(com) for com in rv_com]

        signal_data = pd.DataFrame(index=self.strat_data['vrv_data'].index,
                                   columns=cols)
        self.calc['rv'] = pd.DataFrame(index=self.strat_data['vrv_data'].index,
                                       columns=cols)

        for com in rv_com:
            sig = 'rv_' + str(com)
            self.calc['rv'][com] = (self.strat_data['vrv_data']['returns']
                                    .unstack('ticker')
                                    .ewm(com=com)
                                    .std()
                                    ).stack('ticker') \
                * np.sqrt(constants.trading_days_per_year) * 100.0
            signal_data[sig] = self.calc['rv'][com]

        return signal_data

    def initialize_ts_signals(self, **kwargs):

        ts_com = kwargs.get('ts_com', [0, 1, 5, 10, 21])
        cols = ['ts_' + str(com) for com in ts_com]
        signal_data = pd.DataFrame(index=self.strat_data['vrv_data'].index,
                                   columns=cols)
        for com in ts_com:
            sig = 'ts_' + str(com)
            signal_data[sig] = (self.strat_data['vrv_data']['ts_slope']
                                .unstack('ticker')
                                .ewm(com=com)
                                .mean()
                                ).stack('ticker')

        return signal_data

    def initialize_macro_signals(self, **kwargs):

        tickers = self.get_universe()
        factor_weights = kwargs.get('factor_weights', None)
        macro_factors = kwargs.get('macro_factors', None)

        macro_signal_names = list()
        for i in [0, 1, 2]:
            for j in [0, 1]:
                macro_signal_names.append('vf_' + str(i) + '-mf_' + str(j))

        signal_data = pd.DataFrame(index=self.strat_data['vrv_data'].index,
                                   columns=macro_signal_names)
        for i in factor_weights.columns:
            for j in [0, 1]:

                signal_name = 'vf_' + str(i) + '-mf_' + str(j)

                df = pd.DataFrame(index=macro_factors.index, columns=tickers)
                for ticker in tickers:
                    df[ticker] = macro_factors[j].shift(1) \
                                 * factor_weights.loc[ticker, i]
                df = df.fillna(method='ffill')

                df = df.stack()
                df.index.names = ['date', 'ticker']

                signal_data[signal_name] = df

        for signal in signal_data:
            signal_data[signal] = signal_data[signal].unstack(
                'ticker').fillna(method='ffill').stack('ticker')

        return signal_data

    def compute_signal_z(self, **kwargs):

        signal_data = kwargs.get('signal_data', None)

        signal_data_z = pd.DataFrame(index=signal_data.index,
                                     columns=signal_data.columns)

        signal_data_z = signal_data_z.unstack('ticker')

        for com in signal_data.columns:

            signal_data_com = signal_data[com].unstack('ticker')

            m = signal_data_com.mean(axis=1)
            sd = signal_data_com.std(axis=1)
            signal_data_z[com] = pd.DataFrame(index=signal_data_com.index,
                                              columns=signal_data_com.columns)
            for ticker in signal_data_com.columns:
                signal_data_z[com, ticker] = (signal_data_com[ticker] - m) / sd

        return signal_data_z

    def initialize_portfolio_backtest(self, **kwargs):

        """
        This is core logic for the long/short volatility portfolio backtests
        Regardless of instrument type
        """

        # Convenience
        vsm = self._vsm
        start_date = self.settings['start_date']
        vrv_data = self.strat_data['vrv_data']
        tickers = self.get_universe()

        # Inputs
        signal_data = kwargs.get('signal_data', None)
        signal_name = kwargs.get('signal_name', None)

        # Settings
        trade_frequency_days = self.settings.get('trade_frequency_days')
        quantiles = self.settings.get('quantiles')
        buy_q = self.settings.get('buy_q')
        sell_q = self.settings.get('sell_q')
        trade_tenor_month = self.settings.get('trade_tenor_month')
        hedge_ratio_vov_weight = self.settings.get('hedge_ratio_vov_weight')

        # Template data
        sample_data = vsm.get_data(tickers=['SPY'],
                                   start_date=start_date)
        dates = sample_data.index.get_level_values('date').sort_values()

        # Trade dates
        trade_ind = np.arange(0, len(dates), trade_frequency_days)
        trade_dates = sample_data \
            .iloc[trade_ind] \
            .sort_index(level='date') \
            .index.get_level_values('date')

        maturity_dates = vsm.get_roll_schedule(
            tickers=['SPY'],
            start_date=self.settings['start_date'])['SPY']

        # Stock prices
        stock_prices = md.get_equity_prices(tickers=tickers,
                                            price_field='adj_close',
                                            start_date=start_date) \
                                            ['adj_close']

        # Use the "ideal" positions logic
        signal_ideal_pnl, signal_ideal_positions, signal_pctile \
            = self.compute_signal_quantile_performance(signal_data=signal_data,
                                                       quantiles=quantiles)

        # Identify buys and sells
        # This handles long-only or short-only versions
        if buy_q is not None:
            ix = quantiles.index(buy_q)
            buys = signal_ideal_positions[buy_q][signal_name].loc[trade_dates]

        if sell_q is not None:
            ix = quantiles.index(sell_q)
            sells = signal_ideal_positions[sell_q][signal_name].loc[trade_dates]

        if buy_q is None:
            buys = pd.DataFrame(index=sells.index, columns=sells.columns)
            buys[buys.columns] = 0
        if sell_q is None:
            sells = pd.DataFrame(index=buys.index, columns=buys.columns)
            sells[sells.columns] = 0

        iq_range = quantiles[ix] - quantiles[ix - 1]
        multiplier = iq_range * len(tickers) * trade_tenor_month \
                     * constants.trading_days_per_month / trade_frequency_days

        # Size the trades
        vol_of_vol = vrv_data['vol_of_iv_3m']
        vol_of_vol = vol_of_vol[vol_of_vol.index.get_level_values('date')
                                >= start_date]
        avg_vol_of_vol = vrv_data['vol_of_iv_3m'].mean().mean()
        sizes = ((1 - hedge_ratio_vov_weight) + hedge_ratio_vov_weight
                 * avg_vol_of_vol / vol_of_vol) \
                    .unstack('ticker') / multiplier

        return dates, trade_dates, maturity_dates,\
               stock_prices, buys, sells, sizes

    def compute_portfolio_backtest_options(self, **kwargs):

        # Initialize
        dates, trade_dates, maturity_dates, stock_prices, buys, sells, sizes \
            = self.initialize_portfolio_backtest(**kwargs)

        # Create the security master
        securities = self._create_sec_master_options(
            trade_dates=trade_dates,
            maturity_dates=maturity_dates,
            stock_prices=stock_prices,
            buys=buys,
            sells=sells,
            sizes=sizes,
            **kwargs
        )

        # Create positions
        for date in dates:

            # Create the positions
            x=1

            # Mark the positions
            x=1

            # Create the delta hedges
            x=1

            # Create the daily PNL
            x=1


        # Mark the options
        put_px, call_px = self._price_backtest_positions_options(
            sec_df=securities, stock_prices=stock_prices, **kwargs)

        # Create the positions
        positions = self._create_backtest_positions_options(
            dates=dates, sec_df=securities)

        return positions, securities

    def compute_portfolio_backtest(self, **kwargs):

        # Convenience
        vrv_data = self.strat_data['vrv_data']

        sec_cols = ['instrument',
                    'underlying',
                    'start_date',
                    'maturity_date',
                    'strike',
                    'quantity']

        # Settings
        trade_tenor_month = self.settings.get('trade_tenor_month')
        tc_vega = self.settings.get('transaction_costs_per_unit')

        # Initialize
        dates, trade_dates, maturity_dates, stock_prices, buys, sells, sizes \
            = self.initialize_portfolio_backtest(**kwargs)

        # This part should be in a realized volatility manager
        vrv_data['sq_returns'] = vrv_data['returns'] ** 2.0

        # Create securities
        sec_master = self._create_sec_master(
            trade_dates=trade_dates,
            maturity_dates=maturity_dates,
            buys=buys,
            sells=sells,
            sizes=sizes,
            sec_cols=sec_cols,
            **kwargs
        )

        # Positions
        positions = self._create_backtest_positions(
            dates=dates,
            sec_df=sec_master,
            sec_cols=sec_cols
        )
        first_positions_date = positions.index.get_level_values('date').min()
        dates = dates[dates >= first_positions_date]

        positions = MarkingEngine.mark_volswaps(positions=positions)

        # Now get the total PNL
        cols = ['vega_net', 'vega_gross', 'pnl_gross']
        portfolio_summary = pd.DataFrame(index=dates, columns=cols)
        portfolio_summary['vega_net'] = positions.fillna(value=0) \
            .groupby('date')['quantity'].sum()
        portfolio_summary['vega_gross'] = positions.fillna(value=0)\
            .groupby('date')['quantity_gross'].sum()
        portfolio_summary['pnl_gross'] = positions.groupby('date')[
            'daily_pnl'].sum()

        portfolio_summary['pnl_net'] = portfolio_summary['pnl_gross'] \
            - tc_vega * portfolio_summary['vega_gross'] \
            / trade_tenor_month / constants.trading_days_per_month

        return positions, portfolio_summary, sec_master

    def _create_sec_master_options_date(self,
                                        buys=None,
                                        sells=None,
                                        sizes=None,
                                        trade_date=None,
                                        stock_prices=None,
                                        **kwargs):

        # Settings
        trade_tenor_month = kwargs.get('trade_tenor_month', 3)
        call_deltas_initial_position = kwargs.get(
            'call_deltas_initial_position', [0.40, 0.60])

        # Used for the security master
        sec_cols = ['instrument',
                    'underlying',
                    'trade_date',
                    'maturity_date',
                    'strike',
                    'quantity']

        # Identify the buys and sells
        trade_date_buys = (buys.loc[trade_date][
                               buys.loc[trade_date] > 0]).reset_index()
        trade_date_sells = (sells.loc[trade_date][
                                sells.loc[trade_date] > 0]).reset_index()

        if len(trade_date_buys) == 0:
            return None

        # Create the security master structure
        num_trades = len(trade_date_buys) + len(trade_date_sells)
        trade_ids = np.arange(0, num_trades)
        securities = pd.DataFrame(index=trade_ids, columns=sec_cols)
        securities['trade_date'] = trade_date

        # Underlying
        buy_ind = range(0, len(trade_date_buys))
        sell_ind = range(len(trade_date_buys), num_trades)
        securities.loc[buy_ind, 'underlying'] = trade_date_buys[
            'ticker'].values
        securities.loc[sell_ind, 'underlying'] = trade_date_sells[
            'ticker'].values
        underlyings = [str(s) for s in securities['underlying'].tolist()]

        # Relevant data
        td_data = self._vsm.get_data(tickers=securities['underlying'].tolist(),
                                     start_date=trade_date,
                                     end_date=trade_date) \
            .reset_index(level='date', drop=True)

        sp_data = stock_prices[
            (stock_prices.index.get_level_values('date') == trade_date)
            & (stock_prices.index.get_level_values('ticker').isin(
                underlyings))].reset_index(level='date', drop=True)

        # Traded sizes
        trade_date_sizes = sizes.loc[trade_date]
        securities.loc[buy_ind, 'quantity'] = trade_date_sizes[
            buys.loc[trade_date] > 0].values
        securities.loc[sell_ind, 'quantity'] = -trade_date_sizes[
            sells.loc[trade_date] > 0].values

        # Reset index to underlying in order to map to other data
        securities = securities.reset_index().set_index('underlying',
                                                        drop=True)
        td_data.index.names = ['underlying']

        # Maturity dates
        maturity_fieldname = 'maturity_date_' + str(trade_tenor_month) + 'mc'
        securities['maturity_date'] = td_data[maturity_fieldname]
        tenor_in_days = utils.networkdays(
            start_date=securities['trade_date'],
            end_date=securities['maturity_date'])

        # Now loop over the delta range to get all the individual options
        sec_dict = dict()
        for call_delta in call_deltas_initial_position:
            vols = self._vsm.get_surface_point_by_delta(
                tickers=underlyings,
                call_delta=call_delta,
                tenor_in_days=tenor_in_days,
                start_date=trade_date,
                end_date=trade_date) \
                .reset_index(level='date', drop=True)

            sec_dict[call_delta] = securities.copy()
            sec_dict[call_delta]['ivol_0'] = vols

            # Get moneyness from delta
            # TODO: implement dividends and risk-free rates
            # TODO: better rounding
            sec_dict[call_delta]['strike'] = (
                calcs.black_scholes_moneyness_from_delta(
                    call_delta=call_delta,
                    tenor_in_days=tenor_in_days,
                    ivol=vols / 100.0,
                    risk_free=0,
                    div_yield=0
                ) * sp_data).round(0)

        # Collapse back and overwrite
        securities = pd.concat(sec_dict).reset_index() \
            .rename(columns={'level_0': 'delta_0'})

        # Name the options
        put_ind = securities.index[securities['delta_0'] > 0.50]
        call_ind = securities.index[securities['delta_0'] <= 0.50]
        securities.loc[put_ind, 'instrument'] = \
            securities['underlying'].map(str) + " " \
            + securities['maturity_date'].dt.date.map(str) + " P" \
            + securities['strike'].map(str)
        securities.loc[call_ind, 'instrument'] = \
            securities['underlying'].map(str) + " " \
            + securities['maturity_date'].dt.date.map(str) + " C" \
            + securities['strike'].map(str)
        securities.loc[put_ind, 'option_type'] = 'P'
        securities.loc[call_ind, 'option_type'] = 'C'
        securities = securities.set_index('instrument', drop=True)

        return securities

    def _create_sec_master_options(self,
                                   trade_dates=None,
                                   stock_prices=None,
                                   buys=None,
                                   sells=None,
                                   sizes=None,
                                   **kwargs):

        # Use a dictionary and then concatenate it
        sec_dict_t = dict()

        # Fill forward the VSM data
        self._vsm.data = self._vsm.data\
            .unstack('ticker').fillna(method='ffill',limit=5).stack('ticker')
        stock_prices = stock_prices\
            .unstack('ticker').fillna(method='ffill',limit=5).stack('ticker')

        print('creating security master...')
        for trade_date in trade_dates:
            sec = self._create_sec_master_options_date(buys=buys,
                                                       sells=sells,
                                                       sizes=sizes,
                                                       trade_date=trade_date,
                                                       stock_prices=stock_prices,
                                                       **kwargs)
            if sec is not None:
                sec_dict_t[trade_date] = sec
        sec_df = pd.concat(sec_dict_t)

        # Separate into transactions and securities
        transactions = sec_df.reset_index()[
            ['instrument', 'trade_date', 'quantity']]

        sec_df['instrument_type'] = 'equity_option'

        cols = ['instrument',
                'instrument_type',
                'underlying',
                'option_type',
                'maturity_date',
                'strike']

        security_master = sec_df.reset_index().drop_duplicates('instrument')\
            [cols].set_index('instrument')
        security_master = SecurityMaster(security_master)

        print('getting inputs...')
        start_date = np.min(trade_dates)
        inputs = MarkingEngine.get_instrument_inputs(
            instrument_type='equity_option',
            security_master=security_master,
            start_date=start_date,
            end_date=dt.datetime.today(),
            vsm=self._vsm)

        print('calculating outputs...')
        outputs = MarkingEngine.compute_instrument_outputs(
            instrument_type='equity_option',
            inputs=inputs
        )

        # for current prices
        transactions = pd.merge(
            left=transactions.reset_index(),
            right=outputs.reset_index()[['instrument', 'date', 'price', 'vega']],
            left_on=['instrument', 'trade_date'],
            right_on=['instrument', 'date'])\
            .set_index(['instrument', 'trade_date'])
        del transactions['date']
        del transactions['index']

        # Remember that "quantity" is set in vega terms - let's adjust back
        transactions['quantity_vega'] = transactions['quantity']
        transactions['quantity'] = transactions['quantity_vega']\
                                   / transactions['vega']

        return security_master, transactions, inputs, outputs

    def _create_sec_master(self,
                           trade_dates=None,
                           maturity_dates=None,
                           buys=None,
                           sells=None,
                           sizes=None,
                           sec_cols=None,
                           **kwargs):

        # Settings
        trade_tenor_month = kwargs.get('trade_tenor_month', 3)

        # Convenience
        vrv_data = self.strat_data['vrv_data'].unstack('ticker')

        # Use a dictionary and then concatenate it
        sec_dict = dict()

        c = constants.trading_days_per_month * (trade_tenor_month - 1)

        for trade_date in trade_dates:
            # Buy the on-the-run Nth month
            maturity_date = maturity_dates[
                maturity_dates >= utils.workday(trade_date, c)][0]

            # Identify the buys and sells
            if trade_date in buys.index:
                trade_date_buys = (buys.loc[trade_date][
                                    buys.loc[trade_date] > 0]).reset_index()
            else:
                trade_date_buys = pd.DataFrame(columns=buys.columns)
            if trade_date in sells.index:
                trade_date_sells = (sells.loc[trade_date][
                                    sells.loc[trade_date] > 0]).reset_index()
            else:
                trade_date_sells = pd.DataFrame(columns=sells.columns)

            # Create the security master structure
            num_trades = len(trade_date_buys) + len(trade_date_sells)
            trade_ids = np.arange(0, num_trades)
            securities = pd.DataFrame(index=trade_ids, columns=sec_cols)
            securities['start_date'] = trade_date
            securities['maturity_date'] = maturity_date

            # Underlying
            buy_ind = range(0, len(trade_date_buys))
            sell_ind = range(len(trade_date_buys), num_trades)
            securities.loc[buy_ind, 'underlying'] = trade_date_buys[
                'ticker'].values
            securities.loc[sell_ind, 'underlying'] = trade_date_sells[
                'ticker'].values

            # Traded strikes
            trade_date_vols = vrv_data.loc[trade_date, 'iv_3m']
            securities.loc[buy_ind, 'strike'] = trade_date_vols[
                buys.loc[trade_date] > 0].values
            securities.loc[sell_ind, 'strike'] = trade_date_vols[
                sells.loc[trade_date] > 0].values

            # Traded sizes
            trade_date_sizes = sizes.loc[trade_date]
            securities.loc[buy_ind, 'quantity'] = trade_date_sizes[
                buys.loc[trade_date] > 0].values
            securities.loc[sell_ind, 'quantity'] = -trade_date_sizes[
                sells.loc[trade_date] > 0].values

            securities['instrument'] = \
                securities['underlying'] + " VOLS " \
                + securities['start_date'].dt.date.map(str) + " " \
                + securities['maturity_date'].dt.date.map(str) + " " \
                + securities['strike'].map(str)

            sec_dict[trade_date] = securities

        sec_df = pd.concat(sec_dict).reset_index(drop=True)

        return sec_df

    def _create_backtest_positions_options(self,
                                           security_master=None,
                                           transactions=None,
                                           inputs=None,
                                           outputs=None):

        # Some inputs are replicated from security master
        input_cols = ['spot', 'risk_free', 'dividend_yield', 'iv']
        df = pd.merge(left=inputs[input_cols],
                      right=outputs,
                      left_index=True,
                      right_index=True)

        # Complete blob of all relevant data
        positions = pd.merge(
            left=security_master._data.reset_index(),
            right=df.reset_index(),
            on='instrument')\
            .set_index(['instrument', 'date'])

        # Creating quantity from transactions (very slow method)
        positions['quantity'] = 0.0
        instruments = transactions.index.get_level_values('instrument')
        trade_dates = transactions.index.get_level_values('trade_date')
        position_dates = positions.index.get_level_values('date')
        for i in range(0, len(transactions)):
            if np.mod(i, 100) == 0:
                print(
                'transaction ' + str(i) + ' out of ' + str(len(transactions)))
            sec = security_master._data.loc[instruments[i]]
            maturity_date = sec['maturity_date']
            pos_ind = positions.index[
                (positions.index.get_level_values('instrument') == instruments[
                    i])
                & (position_dates >= trade_dates[i]) & (
                    position_dates <= maturity_date)]
            positions.loc[pos_ind, 'quantity'] += transactions.iloc[i][
                'quantity']

        # Filter down
        positions = positions[positions['quantity'] != 0.0]

        # Market value
        positions['market_value'] = positions['price'] * positions['quantity']

        # PNL
        df = (positions['price'].unstack('instrument')
                .diff(1).fillna(method='ffill')
              * positions['quantity'].unstack('instrument')\
              .shift(1).fillna(method='ffill')) \
            .stack('instrument') \
            .reset_index() \
            .set_index(['instrument', 'date']) \
            .sort_index()[0]

        positions['option_pnl'] = df

        # Delta hedge
        df = \
            (-positions['delta_shares'].unstack('instrument').shift(1)
             * positions['spot'].unstack('instrument').diff(1)
             * positions['quantity'].unstack('instrument')
             ).stack('instrument')
        positions['delta_hedge_pnl'] = df.reset_index() \
            .set_index(['instrument', 'date'])[0]

        # Total hedged pnl
        positions['pnl_gross'] = positions['delta_hedge_pnl'] \
                               + positions['option_pnl']

        cols = ['market_value', 'option_pnl', 'delta_hedge_pnl', 'pnl_gross']
        portfolio_summary = positions[cols].groupby(
            level='date').sum()

        return positions, portfolio_summary

    def _create_backtest_positions(self,
                                   dates=None,
                                   sec_df=None,
                                   sec_cols=None):

        vrv_data = self.strat_data['vrv_data'].unstack('ticker')
        sq_ret = vrv_data['sq_returns']

        port_cols = ['instrument',
                     'date',
                     'quantity',
                     'iv',
                     'rv',
                     'market_value',
                     'daily_pnl']

        pos_dict = dict()

        # Now the daily positions data
        for t in range(0, len(dates)):

            date = dates[t]
            if np.mod(t, 252) == 0:
                print(date)

            sec = sec_df[(sec_df['start_date'] <= date)
                         & (sec_df['maturity_date'] >= date)]

            if len(sec) == 0:
                continue

            pos = pd.DataFrame(index=sec.index, columns=port_cols)
            pos['date'] = date
            pos[sec_cols] = sec[sec_cols]

            # Vol marks (no rolldown for now...)
            iv = vrv_data.loc[date, 'iv_3m'].loc[sec['underlying']]
            pos.index = pos['underlying']
            pos['iv'] = iv
            pos.index = pos.instrument

            # Realized volatility
            vintages = sec[['start_date', 'maturity_date']].drop_duplicates()
            for i in range(0, len(vintages)):

                # Get all swaps with the same start and end date
                sd = vintages.iloc[i]['start_date']
                mtd = vintages.iloc[i]['maturity_date']
                sec_sd = sec[
                    (sec['start_date'] == sd) & (sec['maturity_date'] == mtd)]
                und = sec_sd['underlying'].values.tolist()
                sec_sd.index = sec_sd['instrument']

                # Calculate realized volatility
                df = sq_ret[(sq_ret.index > sd) & (sq_ret.index <= date)][und]
                if len(df) > 0:
                    rv = (df.sum(axis=0) * constants.trading_days_per_year
                          / len(df)) ** 0.5
                    pos.loc[sec_sd.index, 'rv'] = rv.values * 100.0
                else:
                    pos.loc[sec_sd.index, 'rv'] = 0

                # SHOULD be that any NaN's just mean no data yet
                pos['rv'] = pos['rv'].fillna(value=0)

            pos_dict[date] = pos

        positions = pd.concat(pos_dict)
        del positions['instrument']
        positions.index.names = ['date', 'instrument']

        # Forward-fill implied volatility
        positions['iv'] = (positions['iv']
                           .unstack('instrument')
                           .fillna(method='ffill')
                           .stack('instrument'))

        return positions

    def compute_signal_quantile_performance(self, **kwargs):

        """
        This is a highly idealized notion of signal performance involving
        mid-market daily trading... arguably very subject to data noise
        though actually surprisingly comparable to the proper version
        :param kwargs:
        :return:
        """

        vov_com = kwargs.get('vov_com', 63)

        # Methods: 'top_bottom_n', 'top_bottom_pct'
        method = kwargs.get('method', 'top_bottom_q')
        hedge_ratio_vov_weight = kwargs.get('hedge_ratio_vov_weight', 0.50)
        signal_data = kwargs.get('signal_data', None)

        quantiles = kwargs.get('quantiles', [0.0, 0.10, 0.40, 0.60, 0.80, 0.9, 1.0])
        if quantiles[0] > 0:
            quantiles = [0] + quantiles

        # Convenience
        vrv_data = self.strat_data['vrv_data'].unstack('ticker')
        tickers = self.get_universe()

        avg_vol_of_vol = vrv_data['vol_of_iv_3m'].mean().mean()

        signal_pctile = pd.DataFrame(index=signal_data.index,
                                     columns=signal_data.columns)

        signal_pctile = signal_pctile.unstack('ticker')

        for sig in signal_data.columns:

            signal_data_sig = signal_data[sig].unstack('ticker')

            # Cross-sectional percentile
            signal_pctile[sig] = signal_data_sig.rank(axis=1, pct=True)

        signal_positions = dict()
        signal_pnl = dict()

        for i in range(1, len(quantiles)):

            q = quantiles[i]
            q_ = quantiles[i - 1]

            signal_positions[q] = pd.DataFrame(index=signal_data.index,
                                               columns=signal_data.columns) \
                .unstack('ticker')

            signal_pnl[q] = pd.DataFrame(index=signal_data.index,
                                         columns=signal_data.columns) \
                .unstack('ticker')

            for sig in signal_data.columns:
                # Signal positions as quantiles
                signal_positions[q][sig][(signal_pctile[sig].shift(1) > q)] = 0
                signal_positions[q][sig][(signal_pctile[sig].shift(1) < q_)] = 0
                signal_positions[q][sig][(signal_pctile[sig].shift(1) <= q)
                                       & (signal_pctile[sig].shift(1) > q_)] = 1

                signal_pnl[q][sig] = signal_positions[q][sig] * vrv_data['tot_pnl']
                signal_pnl[q][sig] = signal_pnl[q][sig].fillna(value=0)

            # Replace NONE with NAN
            signal_positions[q] = signal_positions[q].where(
                (pd.notnull(signal_positions[q])), 0)

        return signal_pnl, signal_positions, signal_pctile

    def initialize_signals(self, **kwargs):

        rv_iv_signals = self.initialize_rv_iv_signals()
        iv_signals = self.initialize_iv_signals()
        ts_signals = self.initialize_ts_signals()

        signal_data = pd.concat([rv_iv_signals, iv_signals, ts_signals])
        return signal_data

    def compute_master_backtest(self, **kwargs):

        # Required data arguments
        signal_pnl = kwargs.get('signal_pnl', None)
        signal_data = kwargs.get('signal_data', None)

        # Settings
        signals_z_cap = self.settings.get('signals_z_cap', 1.0)
        included_signals = self.settings.get('included_signals',
                                             signal_pnl.columns)
        start_date = self.settings.get('start_date')
        backtest_update_start_date = self.settings.get(
            'backtest_update_start_date')

        # Signals data and PNL
        signal_er = signal_pnl[included_signals].mean() \
                    * constants.trading_days_per_year

        # Single in-sample covariance matrix
        signal_cov = signal_pnl[included_signals].rolling(window=5).sum().cov()\
                     * constants.trading_days_per_year / 5.0
        signal_corr = signal_pnl[included_signals].rolling(
            window=5).sum().corr()

        # Optimization
        optim_output = self.compute_signal_portfolio_optimization(
            signal_er=signal_er,
            signal_corr=signal_corr,
            signal_cov=signal_cov)

        # Combined signal
        combined_signal = pd.DataFrame(index=signal_data.index,
                                       columns=['weighted'])
        combined_signal['weighted'] = 0

        for signal in optim_output['weights'].index:
            sz = self.compute_signal_z(
                signal_data=pd.DataFrame(signal_data[signal]),
                signals_z_cap=signals_z_cap)[signal].stack('ticker')
            combined_signal['weighted'] += sz * \
                optim_output['weights'].loc[signal].values[0]

        positions, portfolio_summary, sec_master = \
            self.compute_portfolio_backtest(
                signal_data=combined_signal,
                signal_name='weighted',
                start_date=start_date,
                backtest_update_start_date=backtest_update_start_date)

        del positions['date']

        return positions, portfolio_summary, sec_master, optim_output

    def compute_signal_backtests(self, **kwargs):

        signal_data = kwargs.get('signal_data', None)
        backtest_update_start_date = kwargs.get('backtest_update_start_date',
                                                None)

        portfolio_summary = dict()

        for signal in signal_data.columns:

            sd = pd.DataFrame(signal_data[signal])

            positions, portfolio_summary[signal], sec_master = \
                self.compute_portfolio_backtest(
                    signal_data=sd,
                    signal_name=signal,
                    backtest_update_start_date=backtest_update_start_date)

        portfolio_summary_df = pd.concat(portfolio_summary)

        return portfolio_summary_df

    def compute_static_strategy(self, **kwargs):
        x=1

    def process_signal_pnl_for_archive(self,
                                       signal_pnl=None,
                                       db_signals=None,
                                       model_param_id=None):

        signal_pnl = signal_pnl.stack()
        signal_pnl.index.names = ['date', 'signal_name']

        df = pd.DataFrame(index=signal_pnl.index,
                          columns=['model_param_id', 'pnl'])
        df['pnl'] = signal_pnl
        df['model_param_id'] = model_param_id
        df['ref_entity_id'] = 'strategy'

        df = pd.merge(left=db_signals,
                      right=df.reset_index(),
                      on='signal_name')

        df = df[['id',
                 'model_param_id',
                 'ref_entity_id',
                 'date',
                 'pnl']]

        return df

    def process_signal_data_for_archive(self,
                                        signal_data=None,
                                        signal_data_z=None,
                                        db_signals=None,
                                        model_param_id=None):

        # This is a portfolio strategy
        # So the signal PNL is at the strategy level
        # But the signals themselves are at the underlying level

        signal_data = signal_data.stack()
        signal_data.index.names = ['date', 'ref_entity_id', 'signal_name']

        signal_data_z = signal_data_z.stack()
        signal_data_z.index.names = ['date', 'ref_entity_id', 'signal_name']

        df = pd.DataFrame(columns=['model_param_id', 'value', 'value_z'])
        df['value'] = signal_data
        df['value_z'] = signal_data_z
        df['model_param_id'] = model_param_id

        df = pd.merge(left=db_signals,
                      right=df.reset_index(),
                      on='signal_name')

        df = df[['id',
                 'model_param_id',
                 'ref_entity_id',
                 'date',
                 'value',
                 'value_z']]

        return df

    def process_model_outputs(self, portfolio_summary=None):

        archive_outputs = dict()
        archive_outputs['pnl_gross'] = portfolio_summary['pnl_gross']
        archive_outputs['pnl_net'] = portfolio_summary['pnl_net']
        archive_outputs['vega_net'] = portfolio_summary['vega_net']
        archive_outputs['vega_gross'] = portfolio_summary['vega_gross']

        df = pd.concat(archive_outputs).reset_index().rename(
            columns={'level_0': 'output_name',
                     0: 'value'})

        return df

    def process_positions_for_archive(self,
                                      positions=None,
                                      db_sec_master=None,
                                      model_id=None,
                                      model_param_id=None):

        if 'date' in positions.columns:
            del positions['date']

        positions = positions.reset_index().rename(
            columns={'instrument': 'instrument_name'})

        cols = ['id', 'instrument_name']
        positions_archive = pd.merge(left=positions.reset_index(),
                                     right=db_sec_master[cols],
                                     on='instrument_name')
        positions_archive = positions_archive.rename(
            columns={'id': 'instrument_id'})
        positions_archive['model_id'] = model_id
        positions_archive['model_param_id'] = model_param_id

        # Build the json column
        position_fields = ['quantity', 'iv', 'rv', 'market_value', 'daily_pnl']
        positions_archive = utils.json_column_from_columns(
            df=positions_archive,
            columns=position_fields,
            new_col_name='value')
        cols = ['model_id', 'model_param_id', 'date', 'instrument_id', 'value']
        positions_archive = positions_archive[cols]

        return positions_archive
