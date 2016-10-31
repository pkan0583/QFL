import pandas as pd
import numpy as np
import datetime as dt
from collections import OrderedDict

import statsmodels.api as sm
import qfl.core.market_data as md
import qfl.core.constants as constants

from qfl.strategies.strategies import Strategy, PortfolioOptimizer


class VegaVsDeltaStrategy(Strategy):

    default_start_date = dt.datetime(2007, 3, 26)

    def __init__(self, **kwargs):

        Strategy.__init__(self, 'equity_vs_vol')

        self.name = 'equity_vs_vol'
        self.strat_data = dict()
        self.calc = dict()

        self.initialize_settings(**kwargs)

    def get_default_settings(self):

        default_settings = OrderedDict()
        default_settings['holding_period_days'] = 1
        default_settings['vol_target_com'] = 63
        default_settings['signals_z_cap'] = 1.0
        default_settings['rolling_beta_com'] = 126
        default_settings['transaction_cost_per_unit'] = 0.05
        default_settings['fd'] = 21
        default_settings['start_date'] = self.default_start_date
        default_settings['vol_fut_series'] = 'VX'
        default_settings['index_fut_series'] = 'ES'
        default_settings['vol_fut_ticker'] = 'VX1'
        default_settings['index_fut_ticker'] = 'ES1'

        return default_settings

    def initialize_settings(self, **kwargs):

        self.settings = dict()
        default_settings = self.get_default_settings()

        # Number of days a position is held for before checking signal
        self.settings['holding_period_days'] = kwargs.get(
            'holding_period_days', default_settings['holding_period_days'])

        # EWM center of mass for trailing window for calculation of strategy
        # PNL volatility used in vol-targeting
        self.settings['vol_target_com'] = kwargs.get(
            'vol_target_com', default_settings['vol_target_com'])

        # Cap for z-scores for individual signals
        self.settings['signals_z_cap'] = kwargs.get(
            'signals_z_cap', default_settings['signals_z_cap'])

        # EWM center of mass for trailing window used for beta calculations
        # Used in hedge ratios
        self.settings['rolling_beta_com'] = kwargs.get(
            'rolling_beta_com', default_settings['rolling_beta_com'])

        # Transaction costs
        self.settings['transaction_cost_per_unit'] \
            = kwargs.get('transaction_cost_per_unit',
                         default_settings['transaction_cost_per_unit'])

        # Futures # days to maturity
        self.settings['fd'] = default_settings['fd']

        # Start date
        self.settings['start_date'] = kwargs.get(
            'start_date', default_settings['start_date'])

        # Futures series for volatility futures
        self.settings['vol_fut_series'] = kwargs.get(
            'vol_fut_series', default_settings['vol_fut_series'])

        # Futures series for index futures
        self.settings['index_fut_series'] = kwargs.get(
            'index_fut_series', default_settings['index_fut_series'])

        # Ticker for volatility futures
        self.settings['vol_fut_ticker'] = self.settings['vol_fut_series'] \
                                          + '1'

        # Ticker for index futures
        self.settings['index_fut_ticker'] = self.settings['index_fut_series'] \
                                            + '1'

    def initialize_data(self, **kwargs):

        # Constant maturity prices
        self.strat_data['cm_vol_fut_prices'] = \
            md.get_constant_maturity_futures_prices_by_series(
                futures_series=self.settings['vol_fut_series'],
                start_date=self.settings['start_date'])['price'] \
                .unstack('days_to_maturity')\
                .reset_index(level='series', drop=True)

        # Rolling VIX futures returns
        self.strat_data['vol_fut_returns'] = \
                md.get_rolling_futures_returns_by_series(
                futures_series=self.settings['vol_fut_series'],
                start_date=self.settings['start_date'],
                level_change=True
            )

        # Rolling index futures returns
        self.strat_data['index_fut_returns'] = \
            md.get_rolling_futures_returns_by_series(
                futures_series=self.settings['index_fut_series'],
                start_date=self.settings['start_date'],
                level_change=False
            )

        # Positioning
        long_field = 'lev_money_positions_long_all'
        short_field = 'lev_money_positions_short_all'
        vol_spec_pos = \
            md.get_cftc_positioning_by_series(
                futures_series=self.settings['vol_fut_series'],
                start_date=self.settings['start_date'])
        vol_spec_pos = vol_spec_pos.set_index('date', drop=True)
        self.strat_data['vol_spec_pos'] = vol_spec_pos[long_field] \
                                        - vol_spec_pos[short_field]
        self.strat_data['vol_fut_open_int'] = vol_spec_pos['open_interest_all']

        index_spec_pos = \
            md.get_cftc_positioning_by_series(
                futures_series=self.settings['index_fut_series'],
                start_date=self.settings['start_date'])
        index_spec_pos = index_spec_pos.set_index('date', drop=True)
        self.strat_data['index_spec_pos'] = index_spec_pos[long_field] \
                                          - index_spec_pos[short_field]
        self.strat_data['index_fut_open_int'] = index_spec_pos['open_interest_all']

        # Hedge ratios
        self.compute_hedge_ratios(rolling_com=self.settings['rolling_beta_com'])
    
    def initialize_signals(self, **kwargs):

        holding_period_days = kwargs.get('holding_period_days', 1)
        vol_target_com = kwargs.get('vol_target_com', 63)
        signals_z_cap = kwargs.get('signals_z_cap', 1.0)

        mom_signals = self.initialize_momentum_signals()
        ts_signals = self.initialize_term_structure_signals()
        pos_signals = self.initialize_positioning_signals()
        rv_signals = self.initialize_rlzd_vs_implied_signals()
        svb_signals = self.initialize_spot_vol_beta_signals()
        c_signals = self.initialize_complex_signals()

        static_pnl = self.compute_static_strategy()

        pos_signal_pnl, pos_positions = self.backtest_signals(
            holding_period_days=holding_period_days,
            signal_data=pos_signals,
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        ts_signal_pnl, ts_positions = self.backtest_signals(
            holding_period_days=holding_period_days,
            signal_data=ts_signals,
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        rv_signal_pnl, rv_positions = self.backtest_signals(
            holding_period_days=holding_period_days,
            signal_data=rv_signals,
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        # Normalize direction
        normalize_direction = True
        if normalize_direction:
            pos_signal_dir = -1.0
            rv_signal_dir = -1.0
            ts_signal_dir = 1.0

            pos_signal_pnl *= pos_signal_dir
            rv_signal_pnl *= rv_signal_dir
            ts_signal_pnl *= ts_signal_dir

            pos_signals *= pos_signal_dir
            rv_signals *= rv_signal_dir
            ts_signals *= ts_signal_dir

        # Complete signal pnl dataset
        signal_pnl = pd.concat([pos_signal_pnl, ts_signal_pnl, rv_signal_pnl],
                               axis=1)
        signal_data = pd.concat([pos_signals, ts_signals, rv_signals], axis=1)
        kwargs['signal_data'] = signal_data
        signal_data_z = self.compute_signal_z(**kwargs)

        output = dict()
        output['signal_data_z'] = signal_data_z
        output['signal_data'] = signal_data
        output['signal_pnl'] = signal_pnl
        output['static_pnl'] = static_pnl

        return output

    def initialize_rlzd_vs_implied_signals(self, **kwargs):

        fd = self.settings['fd']

        dates = self.strat_data['index_fut_returns'].index
        signal_data = pd.DataFrame(index=dates)

        windows = kwargs.get('windows', [10, 21, 42, 63])
        ann_factor = np.sqrt(constants.trading_days_per_year)

        for window in windows:
            signal_data['rv_'+str(window)] = 100.0 * ann_factor * \
                self.strat_data['index_fut_returns'][
                self.settings['index_fut_ticker']].ewm(com=window).std()\
                - self.strat_data['cm_vol_fut_prices'][fd]

        kwargs['signal_data'] = signal_data

        return signal_data

    def initialize_complex_signals(self, **kwargs):

        fd = self.settings['fd']
        sd = 0

        windows = kwargs.get('windows', [1, 5, 10])
        expanding = kwargs.get('expanding', True)
        reg_window = kwargs.get('reg_window', 512)

        dates = self.strat_data['index_fut_returns'].index
        signal_data = pd.DataFrame(index=dates)

        # Term structure slope
        ts_slope = (self.strat_data['cm_vol_fut_prices'][fd]
                    - self.strat_data['cm_vol_fut_prices'][
                        sd]) / self.strat_data['cm_vol_fut_prices'][fd]

        ann_factor = np.sqrt(constants.trading_days_per_year)
        rv = 100.0 * ann_factor * self.strat_data['index_fut_returns'][
            self.settings['index_fut_ticker']].ewm(com=21).std()

        x = pd.DataFrame(index=dates)
        x['ivol'] = self.strat_data['cm_vol_fut_prices'][fd]
        x['ivol_change'] = self.strat_data['cm_vol_fut_prices'][fd].diff(
            1).ewm(com=5).mean()
        x['returns'] = self.strat_data['index_fut_returns'][
            self.settings['index_fut_ticker']].ewm(com=5).mean()
        x['rlzd_implied'] = rv - x['ivol']
        x = x[np.isfinite(x).all(axis=1)]

        if expanding:
            rolling_reg = pd.ols(y=ts_slope, x=x, window_type='expanding')
        else:
            rolling_reg = pd.ols(y=ts_slope, x=x, window=reg_window)

        for window in windows:
            signal_data['c_'+str(window)] = rolling_reg\
                                            .resid\
                                            .ewm(com=window)\
                                            .mean()

        kwargs['signal_data'] = signal_data

        return signal_data
    
    def initialize_spot_vol_beta_signals(self, **kwargs):

        windows = kwargs.get('windows', [10, 21, 42, 63])

        dates = self.strat_data['index_fut_returns'].index
        signal_data = pd.DataFrame(index=dates)

        y = self.strat_data['cm_vol_fut_prices'][self.settings['fd']].diff(1)
        x = self.strat_data['index_fut_returns'][self.settings['index_fut_ticker']]

        for window in windows:
            signal_data['svb_'+str(window)] = \
                y.ewm(com=window).cov(x.ewm(com=window)) \
                / x.ewm(com=window).var()

        kwargs['signal_data'] = signal_data

        return signal_data
    
    def initialize_positioning_signals(self, **kwargs):

        dates = self.strat_data['index_fut_returns'].index
        signal_data = pd.DataFrame(index=dates)
        com = kwargs.get('com', 252)

        signal_data['vol_spec_pos'] = self.strat_data['vol_spec_pos'] \
                                     / self.strat_data['vol_fut_open_int']\
                                     .ewm(com=com).mean()
        signal_data['index_spec_pos'] = self.strat_data['index_spec_pos']\
                                       / self.strat_data['index_fut_open_int']\
                                       .ewm(com=com).mean()

        # Note: these data don't come out until t + x?
        offset = 3
        signal_data = signal_data.shift(offset)

        # Fill to daily
        signal_data = signal_data.fillna(method='ffill')

        # Change signals
        signal_data['vol_spec_pos_chg_s'] = signal_data['vol_spec_pos']\
            .diff(1).ewm(com=21).mean()
        signal_data['index_spec_pos_chg_s'] = signal_data['index_spec_pos'] \
            .diff(1).ewm(com=21).mean()

        signal_data['vol_spec_pos_chg_l'] = signal_data['vol_spec_pos']\
            .diff(1).ewm(com=63).mean()
        signal_data['index_spec_pos_chg_l'] = signal_data['index_spec_pos'] \
            .diff(1).ewm(com=63).mean()

        kwargs['signal_data'] = signal_data

        return signal_data
    
    def initialize_term_structure_signals(self, **kwargs):

        fd = self.settings['fd']
        sd = 0

        # Looking at how long to smooth term structure
        windows = kwargs.get('windows', [0, 1, 5, 10])

        dates = self.strat_data['index_fut_returns'].index
        signal_data = pd.DataFrame(index=dates)

        ts_slope = (self.strat_data['cm_vol_fut_prices'][fd]
                    - self.strat_data['cm_vol_fut_prices'][sd]) / \
                      self.strat_data['cm_vol_fut_prices'][fd]

        for window in windows:
            signal_data['ts_'+str(window)] = ts_slope.ewm(com=window).mean()

        kwargs['signal_data'] = signal_data

        return signal_data

    def initialize_momentum_signals(self, **kwargs):

        # Studying impact of various time windows for prior performance
        windows = kwargs.get('windows', [1, 5, 10, 21])

        # Get the static backtest data just to be sure
        backtest_data = self.compute_static_strategy(**kwargs)

        dates = self.strat_data['index_fut_returns'].index
        signal_data = pd.DataFrame(index=dates)
        for window in windows:
            signal_data['mom_'+str(window)] = backtest_data['static_pnl'] \
                .ewm(com=window).mean()

        kwargs['signal_data'] = signal_data

        return signal_data

    def compute_hedge_ratios(self, **kwargs):

        fd = self.settings['fd']
        rolling_beta_com = kwargs.get('rolling_beta_com', 21)

        # Trailing betas
        y = self.strat_data['cm_vol_fut_prices'][fd].diff(1)

        # Predict volatility move as a function of stuff
        x = pd.DataFrame(index=self.strat_data['index_fut_returns'].index)
        x['index_return'] = self.strat_data['index_fut_returns'][
            self.settings['index_fut_ticker']]
        x['vol_level'] = self.strat_data['cm_vol_fut_prices'][fd].shift(1)
        x['interaction'] = x['index_return'] * x['vol_level']
        exog_vars = ['index_return', 'vol_level', 'interaction']

        r1 = pd.ols(y=y, x=x)
        r2 = pd.ols(y=np.log(r1.resid ** 2), x=x[exog_vars])
        pred_sq_err = np.exp(r2.y_fitted)

        weights = 1. / pred_sq_err
        reg_data = x
        reg_data['endog'] = y
        reg_data['weights'] = weights

        reg_data = reg_data[np.isfinite(reg_data).all(axis=1)]
        reg_data = sm.add_constant(reg_data)

        r3 = sm.WLS(endog=reg_data['endog'],
                    exog=reg_data[['const'] + exog_vars],
                    weights=reg_data['weights']).fit()

        vol_level_grid = np.arange(10, 51)
        beta_df = pd.DataFrame(index=vol_level_grid, columns=['beta'])
        for i in range(0, len(vol_level_grid)):
            beta_df.loc[vol_level_grid[i], 'beta'] \
                = (1.0 / 100.0) * (r3.params.index_return +
                                   vol_level_grid[i] * r3.params.interaction)

        # Historical spot-vol betas (vol points per 1%)
        self.calc['spot_vol_beta'] = (r3.params.index_return
                                  + x['vol_level']
                                  * r3.params.interaction) / 100.0

        self.calc['rolling_spot_vol_beta'] = \
            y.ewm(com=rolling_beta_com).cov(x['index_return']
                .ewm(com=rolling_beta_com)) \
            / x['index_return'].ewm(com=rolling_beta_com).var()

    def compute_static_strategy(self, **kwargs):

        # Vega-vs-delta (short vol, short stocks)
        self.calc['backtest_data'] \
            = pd.DataFrame(index=self.strat_data['index_fut_returns'].index)

        self.calc['backtest_data']['vega'] \
            = self.strat_data['vol_fut_returns'][self.settings['vol_fut_ticker']]

        self.calc['backtest_data']['delta'] \
            = self.strat_data['index_fut_returns'][self.settings['index_fut_ticker']]\
            * 100.0 * self.calc['spot_vol_beta']

        self.calc['backtest_data']['static_pnl'] \
            = self.calc['backtest_data']['delta']\
            - self.calc['backtest_data']['vega']

        return self.calc['backtest_data']

    def backtest_signals(self, **kwargs):

        dates = self.strat_data['index_fut_returns'].index
        holding_period_days = kwargs.get('holding_period_days', 1)
        vol_target_com = kwargs.get('vol_target_com', None)

        signal_data_z = self.compute_signal_z(**kwargs)

        backtest_signals = signal_data_z.columns
        signal_pnl = pd.DataFrame(index=dates,
                                  columns=backtest_signals)

        positions = signal_data_z.shift(holding_period_days)
        backtest_data = self.compute_static_strategy(**kwargs)

        # Volatility target the whole strategy vs the static backtest
        if vol_target_com is not None:

            static_strategy_vol = backtest_data['static_pnl']\
                .ewm(com=vol_target_com).std()
            static_strategy_min_vol = static_strategy_vol \
                .iloc[vol_target_com:].min()
            for col in backtest_signals:
                positions[col] = (positions[col] * static_strategy_min_vol
                                / static_strategy_vol)

        # PNL
        backtest_data.index = pd.to_datetime(backtest_data.index)
        positions.index = pd.to_datetime(positions.index)

        for col in backtest_signals:
            signal_pnl[col] = backtest_data['static_pnl'] * positions[col]

        return signal_pnl, positions
