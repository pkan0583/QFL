import pandas as pd
import numpy as np
import datetime as dt
import struct

import statsmodels.api as sm
import qfl.core.market_data as md
import qfl.core.constants as constants
import qfl.utilities.basic_utilities as utils
import qfl.core.calcs as calcs

from qfl.strategies.strategies import Strategy, PortfolioStrategy, PortfolioOptimizer
from qfl.core.market_data import VolatilitySurfaceManager


class VolswapRvStrategy(PortfolioStrategy):

    # volatility surface manager
    _vsm = None
    _universe = None
    strat_data = struct
    calc = struct
    settings = struct

    def __init__(self):

        # Call super
        PortfolioStrategy.__init__(self)

        # Default universe
        self.set_universe()

        # Beginning of ORATS data
        self.settings.default_start_date = dt.datetime(2010, 1, 3)

    def set_universe(self, **kwargs):
        self._universe = md.get_etf_vol_universe()

    def clean_data(self, **kwargs):

        x=1

    def initialize_data(self, **kwargs):

        self._vsm = kwargs.get('volatility_surface_manager', None)
        self.settings.start_date = kwargs.get('start_date',
                                              self.settings.default_start_date)
        start_date = self.settings.start_date

        # Set universe
        tickers = self.get_universe(**kwargs)

        # Implied volatility data
        if self._vsm is None:
            self._vsm = VolatilitySurfaceManager()
            if self._vsm.data is None:
                self._vsm.load_data(tickers=tickers,
                                    start_date=start_date)
        fields = ['iv_2m',
                  'iv_3m',
                  'days_to_maturity_1mc',
                  'days_to_maturity_2mc',
                  'days_to_maturity_3mc']

        ivol = self._vsm.get_data(tickers=tickers,
                                  fields=fields,
                                  start_date=start_date)
        ivol = ivol.unstack(level='ticker')

        # Stock price data
        stock_price_start_date = utils.workday(
            date=start_date,
            num_days=-constants.trading_days_per_year * 2)
        stock_prices = md.get_equity_prices(tickers=tickers,
                                            start_date=stock_price_start_date)
        stock_prices = stock_prices['adj_close'].unstack(level='ticker')

        # Cleaning implied volatility data
        deg_f = 5
        res_com = 5
        clean_ivol = pd.DataFrame(index=ivol.index, columns=ivol.columns)

        # clean_ivol['iv_2m'], normal_tests_2m = calcs.clean_implied_vol_data(
        #     tickers=tickers,
        #     stock_prices=stock_prices,
        #     ivol=ivol['iv_2m'],
        #     ref_ivol_ticker='SPY',
        #     deg_f=deg_f,
        #     res_com=res_com
        # )
        #
        # clean_ivol['iv_3m'], normal_tests_3m = calcs.clean_implied_vol_data(
        #     tickers=tickers,
        #     stock_prices=stock_prices,
        #     ivol=ivol['iv_3m'],
        #     ref_ivol_ticker='SPY',
        #     deg_f=deg_f,
        #     res_com=res_com
        # )
        #
        # # For now we're just going to throw out anything with stat < 100
        # tmp2 = normal_tests_2m[normal_tests_2m['stat'] > 10.0].index.tolist()
        # tmp3 = normal_tests_3m[normal_tests_3m['stat'] > 10.0].index.tolist()
        # final_tickers = list(set(tmp2).intersection(set(tmp3)))

        # # Filter down (sort of awkward, is there a better way?)
        # vrv_data = clean_ivol.copy(deep=True)
        # for i in range(len(vrv_data.columns)-1, 0, -1):
        #     if vrv_data.columns[i][1] not in final_tickers:
        #         del vrv_data[vrv_data.columns[i]]

        clean_ivol = ivol
        vrv_data = ivol.copy(deep=True)
        final_tickers = tickers

        mat_cols =['days_to_maturity_1mc',
                   'days_to_maturity_2mc',
                   'days_to_maturity_3mc']
        for col in mat_cols:
            vrv_data[col] = ivol[col]

        # Add term structure slope
        stock_prices = stock_prices.stack('ticker')
        vrv_data = vrv_data.stack('ticker')
        vrv_data['ts_slope'] = vrv_data['iv_3m'] - vrv_data['iv_2m']
        vrv_data['stock_prices'] = stock_prices

        # Data storage
        self.strat_data.clean_ivol = clean_ivol
        self.strat_data.raw_ivol = ivol
        self.strat_data.vrv_data = vrv_data
        self.strat_data.stock_prices = stock_prices
        self.settings.tickers = final_tickers

    def get_roll_schedule(self, **kwargs):

        # For now we're going to assume that all names have the same roll sched

        tickers = ['SPY']
        maturity_dates = self._vsm.get_roll_schedule(
            tickers=tickers,
            start_date=self.settings.start_date)

        return maturity_dates

    def process_data(self, **kwargs):

        vrv_data = self.strat_data.vrv_data

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

        vrv_data = vrv_data.unstack('ticker')

        self.vrv_data = vrv_data

    def initialize_tick_rv_signals(self, **kwargs):

        diff_in_pct = kwargs.get('diff_in_pct', True)

        tickers = self.get_universe()
        tick_vol = md.get_equity_tick_realized_volatility(
            tickers=tickers,
            start_date=self.settings.start_date)

        windows = [10, 20, 60, 120, 252]
        cols = [str('tick_diff_') + str(w) for w in windows]
        signal_data = pd.DataFrame(index=self.strat_data.vrv_data.index,
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

        signal_data = pd.DataFrame(index=self.strat_data.vrv_data.index,
                                   columns=cols)
        self.calc.rv = pd.DataFrame(index=self.strat_data.vrv_data.index,
                                    columns=cols)

        for com in rv_com:
            sig = 'rv_iv_' + str(com)
            self.calc.rv[com] = (self.strat_data.vrv_data['returns']
                                .unstack('ticker')
                                .ewm(com=com)
                                .std()
                                ).stack('ticker') \
                * np.sqrt(constants.trading_days_per_year) * 100.0
            signal_data[sig] = self.calc.rv[com] - self.strat_data.vrv_data['iv_3m']
            if diff_in_pct:
                signal_data[sig] /= self.strat_data.vrv_data['iv_3m']

        return signal_data

    def initialize_iv_signals(self, **kwargs):

        iv_com = kwargs.get('iv_com', [10, 21, 42, 63, 126, 252])
        cols = ['iv_' + str(com) for com in iv_com]
        diff_in_pct = kwargs.get('diff_in_pct', True)

        signal_data = pd.DataFrame(index=self.strat_data.vrv_data.index,
                                   columns=cols)

        self.calc.iv_ewm = pd.DataFrame(index=self.strat_data.vrv_data.index,
                                        columns=cols)

        for com in iv_com:
            sig = 'iv_' + str(com)
            self.calc.iv_ewm[com] = (self.strat_data.vrv_data['iv_3m']
                                    .unstack('ticker')
                                    .ewm(com=com)
                                    .mean()
                                    ).stack('ticker')
            signal_data[sig] = self.strat_data.vrv_data['iv_3m'] - self.calc.iv_ewm[com]
            if diff_in_pct:
                signal_data[sig] /= self.strat_data.vrv_data['iv_3m']

        return signal_data

    def initialize_rv_signals(self, **kwargs):

        rv_com = kwargs.get('rv_com', [10, 21, 42, 63, 126, 252])
        cols = ['rv_' + str(com) for com in rv_com]

        signal_data = pd.DataFrame(index=self.strat_data.vrv_data.index,
                                   columns=cols)
        self.calc.rv = pd.DataFrame(index=self.strat_data.vrv_data.index,
                                    columns=cols)

        for com in rv_com:
            sig = 'rv_' + str(com)
            self.calc.rv[com] = (self.strat_data.vrv_data['returns']
                                .unstack('ticker')
                                .ewm(com=com)
                                .std()
                                ).stack('ticker') \
                * np.sqrt(constants.trading_days_per_year) * 100.0
            signal_data[sig] = self.calc.rv[com]

        return signal_data

    def initialize_ts_signals(self, **kwargs):

        ts_com = kwargs.get('ts_com', [0, 1, 5, 10, 21])
        cols = ['ts_' + str(com) for com in ts_com]
        signal_data = pd.DataFrame(index=self.strat_data.vrv_data.index,
                                   columns=cols)
        for com in ts_com:
            sig = 'ts_' + str(com)
            signal_data[sig] = (self.strat_data.vrv_data['ts_slope']
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

        signal_data = pd.DataFrame(index=self.strat_data.vrv_data.index,
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

    def compute_realistic_portfolio_backtest_options(self, **kwargs):

        # Convenience
        vsm = self._vsm
        start_date = self.settings.start_date
        vrv_data = self.strat_data.vrv_data

        sec_cols = ['instrument',
                    'underlying',
                    'start_date',
                    'maturity_date',
                    'strike',
                    'quantity']

        tickers = self.get_universe()

        # Settings
        trade_frequency_days = kwargs.get('trade_frequency_days', 5)
        quantiles = kwargs.get('quantiles', [0.10, 0.30, 0.70, 0.90, 1.0])
        buy_q = kwargs.get('buy_q', 0.90)
        sell_q = kwargs.get('sell_q', 0.30)
        trade_tenor_month = kwargs.get('trade_tenor_month', 3)
        hedge_ratio_vov_weight = kwargs.get('hedge_ratio_vov_weight', 0.50)
        signal_data = kwargs.get('signal_data', None)
        signal_name = kwargs.get('signal_name', None)
        call_deltas_initial_position = kwargs.get(
            'call_deltas_initial_position', [0.40, 0.60])
        relative_sizes_initial_position = kwargs.get(
            'relative_sizes_initial_position', [0.40, 0.60])

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
            start_date=self.settings.start_date)['SPY']

        # Stock prices
        stock_prices = md.get_equity_prices(tickers=tickers,
                                            price_field='adj_close',
                                            start_date=start_date) \
                                            ['adj_close']

        sec_cols = ['instrument',
                    'underlying',
                    'trade_date',
                    'maturity_date',
                    'strike',
                    'quantity']

        # Use the "ideal" positions logic
        signal_ideal_pnl, signal_ideal_positions, signal_pctile \
            = self.compute_signal_quantile_performance(signal_data=signal_data,
                                                       quantiles=quantiles)

        # Identify buys and sells
        buys = signal_ideal_positions[buy_q][signal_name].loc[trade_dates]
        sells = signal_ideal_positions[sell_q][signal_name].loc[trade_dates]

        # Size the trades
        avg_vol_of_vol = vrv_data['vol_of_iv_3m'].mean().mean()
        sizes = ((1 - hedge_ratio_vov_weight) + hedge_ratio_vov_weight
                 * avg_vol_of_vol / vrv_data['vol_of_iv_3m']) \
            .unstack('ticker')

        sec_df = self._create_backtest_sec_master_options(
            trade_dates=trade_dates,
            maturity_dates=maturity_dates,
            stock_prices=stock_prices,
            buys=buys,
            sells=sells,
            sizes=sizes,
            sec_cols=sec_cols,
            **kwargs
        )

    def compute_realistic_portfolio_backtest(self, **kwargs):

        # Convenience
        vsm = self._vsm
        start_date = self.settings.start_date
        vrv_data = self.strat_data.vrv_data

        sec_cols = ['instrument',
                    'underlying',
                    'start_date',
                    'maturity_date',
                    'strike',
                    'quantity']

        # Settings
        trade_frequency_days = kwargs.get('trade_frequency_days', 5)
        quantiles = kwargs.get('quantiles', [0.10, 0.30, 0.70, 0.90, 1.0])
        buy_q = kwargs.get('buy_q', 0.90)
        sell_q = kwargs.get('sell_q', 0.30)
        trade_tenor_month = kwargs.get('trade_tenor_month', 3)
        hedge_ratio_vov_weight = kwargs.get('hedge_ratio_vov_weight', 0.50)
        signal_data = kwargs.get('signal_data', None)
        signal_name = kwargs.get('signal_name', None)

        # Template data
        sample_data = vsm.get_data(tickers=['SPY'],
                                   start_date=start_date)
        dates = sample_data.index.get_level_values('date').sort_values()

        # Maturity dates
        maturity_dates = vsm.get_roll_schedule(tickers=['SPY'],
                                               start_date=start_date)['SPY']

        # Trade dates
        trade_ind = np.arange(0, len(dates), trade_frequency_days)
        trade_dates = sample_data \
            .iloc[trade_ind] \
            .sort_index(level='date') \
            .index.get_level_values('date')

        # Use the "ideal" positions logic
        signal_ideal_pnl, signal_ideal_positions, signal_pctile \
            = self.compute_signal_quantile_performance(signal_data=signal_data,
                                                       quantiles=quantiles)

        # Identify buys and sells
        buys = signal_ideal_positions[buy_q][signal_name].loc[trade_dates]
        sells = signal_ideal_positions[sell_q][signal_name].loc[trade_dates]

        # Size the trades
        avg_vol_of_vol = vrv_data['vol_of_iv_3m'].mean().mean()
        sizes = ((1 - hedge_ratio_vov_weight) + hedge_ratio_vov_weight
                 * avg_vol_of_vol / vrv_data['vol_of_iv_3m']) \
                .unstack('ticker')

        # This part should be in a realized volatility manager
        vrv_data['sq_returns'] = vrv_data['returns'] ** 2.0

        # Create securities
        sec_df = self._create_backtest_sec_master(
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
            sec_df=sec_df,
            sec_cols=sec_cols
        )

        # TODO: I should really move the marking logic out, it's pretty general

        # Now mark the trades
        positions['market_value'] \
            = calcs.volswap_market_value(
                iv=positions['iv'],
                rv=positions['rv'],
                strike=positions['strike'],
                maturity_date=positions['maturity_date'],
                start_date=positions['start_date'],
                date=positions['date']
            )

        # Now get the PNL
        positions['daily_pnl'] = (
            positions.unstack('instrument').quantity.shift(1)
                * (positions.unstack('instrument').market_value
                - positions.unstack('instrument').market_value.shift(1))
            ).stack('instrument')

        # Gross position
        positions['gross_quantity'] = positions['quantity'].abs()

        # Now get the total PNL
        cols = ['net_vega', 'gross_vega', 'total_pnl']
        portfolio_summary = pd.DataFrame(index=dates, columns=cols)
        portfolio_summary['net_vega'] = positions.fillna(value=0) \
            .groupby('date')['quantity'].sum()
        portfolio_summary['gross_vega'] = positions.fillna(value=0)\
            .groupby('date')['gross_quantity'].sum()
        portfolio_summary['total_pnl'] = positions.groupby('date')[
            'daily_pnl'].sum()
        portfolio_summary['cum_pnl'] = portfolio_summary['total_pnl'].cumsum()

        return positions, portfolio_summary

    def _create_backtest_sec_master_options(self,
                                            trade_dates=None,
                                            maturity_dates=None,
                                            stock_prices=None,
                                            buys=None,
                                            sells=None,
                                            sizes=None,
                                            sec_cols=None,
                                            **kwargs):

        # Settings
        trade_tenor_month = kwargs.get('trade_tenor_month', 3)
        hedge_ratio_vov_weight = kwargs.get('hedge_ratio_vrv_weight', 0.50)
        call_deltas_initial_position = kwargs.get(
            'call_deltas_initial_position', None)

        # Convenience
        vrv_data = self.strat_data.vrv_data.unstack('ticker')

        # Size the trades
        avg_vol_of_vol = vrv_data['vol_of_iv_3m'].mean().mean()
        sizes = ((1 - hedge_ratio_vov_weight) + hedge_ratio_vov_weight
                 * avg_vol_of_vol / vrv_data['vol_of_iv_3m']) \
                 .unstack('ticker')

        # Use a dictionary and then concatenate it
        sec_dict_t = dict()

        for trade_date in trade_dates:

            # Identify the buys and sells
            trade_date_buys = (buys.loc[trade_date][
                                   buys.loc[trade_date] > 0]).reset_index()
            trade_date_sells = (sells.loc[trade_date][
                                    sells.loc[trade_date] > 0]).reset_index()

            # Create the security master structure
            num_trades = len(trade_date_buys) + len(trade_date_sells)
            trade_ids = np.arange(0, num_trades)
            securities = pd.DataFrame(index=trade_ids, columns=sec_cols)
            securities['start_date'] = trade_date

            # Underlying
            buy_ind = range(0, len(trade_date_buys))
            sell_ind = range(len(trade_date_buys), num_trades)
            securities.loc[buy_ind, 'underlying'] = trade_date_buys[
                'ticker'].values
            securities.loc[sell_ind, 'underlying'] = trade_date_sells[
                'ticker'].values
            underlyings = [str(s) for s in securities['underlying'].tolist()]

            # Relevant data
            td_data = self.vsm.get_data(tickers=securities['underlying'].tolist(),
                                        start_date=trade_date,
                                        end_date=trade_date) \
                .reset_index(level='date', drop=True)

            sp_data = stock_prices[
                (stock_prices.index.get_level_values('date') == trade_date)
                & (stock_prices.index.get_level_values('ticker').isin(underlyings))
                ].reset_index(level='date', drop=True)

            # Traded sizes
            trade_date_sizes = sizes.loc[trade_date]
            securities.loc[buy_ind, 'quantity'] = trade_date_sizes[
                buys.loc[trade_date] > 0].values
            securities.loc[sell_ind, 'quantity'] = -trade_date_sizes[
                sells.loc[trade_date] > 0].values

            # Reset index to underlying in order to map to other data
            securities = securities.reset_index('underlying', drop=True)

            # Maturity dates
            maturity_fieldname = 'maturity_date_' + str(trade_tenor_month) + 'mc'
            securities['maturity_date'] = td_data[maturity_fieldname]
            tenor_in_days = utils.networkdays(
                start_date=securities['trade_date'],
                end_date=securities['maturity_date'])

            # Now loop over the delta range to get all the individual options
            sec_dict = dict()
            for call_delta in call_deltas_initial_position:
                vols = self.vsm.get_surface_point(tickers=underlyings,
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
                    ) * sp_data).round(2)

            # Collapse back and overwrite
            securities = pd.concat(sec_dict).reset_index()\
                        .rename(columns={'level_0':'delta_0'})

            # Name the options
            put_ind = securities.index[securities['delta_0'] < 0.50]
            call_ind = securities.index[securities['delta_0'] >= 0.50]
            securities.loc[put_ind, 'instrument'] = \
                securities['underlying'].map(str) + " " \
                + securities['maturity_date'].dt.date.map(str) + " P" \
                + securities['strike'].map(str)
            securities.loc[call_ind, 'instrument'] = \
                securities['underlying'].map(str) + " " \
                + securities['maturity_date'].dt.date.map(str) + " C" \
                + securities['strike'].map(str)
            securities = securities.set_index('instrument', drop=True)

            sec_dict_t[trade_date] = securities

        sec_df = pd.concat(sec_dict_t)
        return sec_df

    def _create_backtest_sec_master(self,
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
        vrv_data = self.strat_data.vrv_data.unstack('ticker')

        # Use a dictionary and then concatenate it
        sec_dict = dict()

        c = constants.trading_days_per_month * (trade_tenor_month - 1)

        for trade_date in trade_dates:
            # Buy the on-the-run Nth month
            maturity_date = maturity_dates[
                maturity_dates >= utils.workday(trade_date, c)][0]

            # Identify the buys and sells
            trade_date_buys = (buys.loc[trade_date][
                                   buys.loc[trade_date] > 0]).reset_index()
            trade_date_sells = (sells.loc[trade_date][
                                    sells.loc[trade_date] > 0]).reset_index()

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
                                           dates=None,
                                           sec_df=None,
                                           sec_cols=None):

        vrv_data = self.strat_data.vrv_data.unstack('ticker')
        sq_ret = vrv_data['sq_returns']

        port_cols = ['instrument',
                     'date',
                     'quantity',
                     'tenor_in_days',
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

            sec = sec_df[(sec_df['trade_date'] <= date)
                       & (sec_df['maturity_date'] >= date)]

            if len(sec) == 0:
                continue

            pos = pd.DataFrame(index=sec.index, columns=port_cols)
            pos['date'] = date
            pos[sec_cols] = sec[sec_cols]

            pos['tenor_in_days'] = utils.networkdays(
                start_date=sec['trade_date'],
                end_date=sec['maturity_date'])

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

    def _create_backtest_positions(self,
                                   dates=None,
                                   sec_df=None,
                                   sec_cols=None):

        vrv_data = self.strat_data.vrv_data.unstack('ticker')
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

        quantiles = kwargs.get('quantiles', [0.0, 0.20, 0.40, 0.60, 0.80, 1.0])
        if quantiles[0] > 0:
            quantiles = [0] + quantiles

        # Convenience
        vrv_data = self.strat_data.vrv_data
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
        vrv_data = vrv_data.unstack('ticker')

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

        rv_iv_signals = self.initialize_rv_iv_signals()
        iv_signals = self.initialize_iv_signals()
        ts_signals = self.initialize_ts_signals()

        rv_iv_pnl, rv_iv_pos, rv_iv_pctile \
            = self.compute_signal_quantile_performance(signal_data=rv_iv_signals)

        iv_pnl, iv_pos, iv_pctile \
            = self.compute_signal_quantile_performance(signal_data=iv_signals)

        ts_pnl, ts_pos, ts_pctile \
            = self.compute_signal_quantile_performance(signal_data=ts_signals)

        # TODO: turn this whole thing into a big dataframe
        signal_pnl = dict()
        signal_pos = dict()
        signal_data = pd.concat([rv_iv_signals, iv_signals, ts_signals])
        signal_pctile = pd.concat([rv_iv_pctile, iv_pctile, ts_pctile])
        for q in rv_iv_pnl.keys():
            signal_pnl[q] = pd.concat([rv_iv_pnl[q], iv_pnl[q], ts_pnl[q]])
            signal_pos[q] = pd.concat([rv_iv_pos[q], iv_pos[q], ts_pos[q]])

        return signal_pnl, signal_pos, signal_pctile

    def compute_static_strategy(self, **kwargs):
        x=1