import pandas as pd
import numpy as np
import datetime as dt
import struct

import statsmodels.api as sm
import qfl.core.market_data as md
import qfl.core.constants as constants

from qfl.strategies.strategies import Strategy


class VixCurveStrategy(Strategy):

    def __init__(self):

        Strategy.__init__(self, 'VixCurve')

        self.strat_data = struct
        self.calc = struct
        self.settings = struct

        # Beginning of semi-clean VIX futures data
        self.settings.default_start_date = dt.datetime(2007, 3, 26)

    def initialize_data(self, **kwargs):

        # Settings
        self.settings.index_fut_series = kwargs.get('index_futures_series','ES')
        self.settings.vol_fut_series = kwargs.get('vol_futures_series', 'VX')

        short_month = kwargs.get('short_month', 1)
        long_month = kwargs.get('long_month', 5)
        self.settings.fd = [21 * short_month, 21 * long_month]
        self.settings.short_ticker = self.settings.vol_fut_series \
                                   + str(short_month)
        self.settings.long_ticker = self.settings.vol_fut_series \
                                  + str(long_month)
        self.settings.index_fut_ticker = self.settings.index_fut_series + "1"

        self.settings.start_date = kwargs.get('start_date',
                                              self.settings.default_start_date)

        # Constant maturity prices
        self.strat_data.cm_vol_fut_prices = \
            md.get_constant_maturity_futures_prices_by_series(
                futures_series=self.settings.vol_fut_series,
                start_date=self.settings.start_date)['price'] \
                .unstack('days_to_maturity')\
                .reset_index(level='series', drop=True)

        # Rolling returns
        self.strat_data.vol_fut_returns = md.get_rolling_futures_returns_by_series(
            futures_series=self.settings.vol_fut_series,
            start_date=self.settings.start_date,
            level_change=True
        )

        # CM returns
        self.strat_data.cm_vol_fut_returns = self.strat_data.cm_vol_fut_prices.diff(1)

        # Positioning
        vol_spec_pos = \
            md.get_cftc_positioning_by_series(
                futures_series=self.settings.vol_fut_series,
                start_date=self.settings.start_date)
        vol_spec_pos = vol_spec_pos.set_index('date', drop=True)
        self.strat_data.vol_spec_pos = vol_spec_pos['lev_money_positions_long_all'] \
                                       - vol_spec_pos['lev_money_positions_short_all']
        self.strat_data.vol_fut_open_int = vol_spec_pos['open_interest_all']

        # Index data for realized vol
        self.strat_data.index_fut_returns = md.get_rolling_futures_returns_by_series(
            futures_series=self.settings.index_fut_series,
            start_date=self.settings.start_date,
            level_change=False
        )

    def compute_hedge_ratios(self, **kwargs):

        rolling_beta_com = kwargs.get('rolling_beta_com', 63)

        # Predict long term volatility move as a function of short
        fd = self.settings.fd
        y = self.strat_data.cm_vol_fut_returns[fd[1]]
        x = pd.DataFrame(index=self.strat_data.cm_vol_fut_returns.index)
        x['front_return'] = self.strat_data.cm_vol_fut_returns[fd[0]]
        x['front_level'] = self.strat_data.cm_vol_fut_prices[fd[0]].shift(1)
        x['interaction'] = x['front_return'] * x['front_level']
        exog_vars = ['front_return', 'front_level', 'interaction']

        # OLS to start
        r1 = pd.ols(y=y, x=x)

        # Now use OLS residuals for WLS
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

        # Historical front/back betas
        self.calc.front_back_beta = (r3.params.front_return
                                  + x['front_level']
                                  * r3.params.interaction)

        self.calc.rolling_front_back_beta = \
            y.ewm(com=rolling_beta_com).cov(x['front_return']
                                            .ewm(com=rolling_beta_com)) \
            / x['front_return'].ewm(com=rolling_beta_com).var()

        buffer = 21
        self.calc.rolling_front_back_beta.iloc[0:buffer] \
            = self.calc.rolling_front_back_beta.iloc[buffer]

        self.calc.r1 = r1
        self.calc.r2 = r2
        self.calc.r3 = r3

    def compute_static_strategy(self, **kwargs):

        fd = self.settings.fd
        st = self.settings.short_ticker
        lt = self.settings.long_ticker

        # Vega-vs-delta (short vol, short stocks)
        self.calc.backtest_data \
            = pd.DataFrame(index=self.strat_data.vol_fut_returns.index)

        self.calc.backtest_data['front'] = self.strat_data.vol_fut_returns[st]
        self.calc.backtest_data['back'] = self.strat_data.vol_fut_returns[lt] \
            / self.calc.rolling_front_back_beta

        self.calc.backtest_data['static_pnl'] \
            = self.calc.backtest_data['front'] - self.calc.backtest_data['back']

        return self.calc.backtest_data

    def initialize_rel_ts_signals(self, **kwargs):

        # Idea here is, how steep is the curve conditional on its level?

        fd = self.settings.fd
        sd = 0

        windows = kwargs.get('windows', [1, 5, 10])
        expanding = kwargs.get('expanding', True)
        reg_window = kwargs.get('reg_window', 512)
        pct_ts = kwargs.get('pct_ts', False)

        dates = self.strat_data.cm_vol_fut_prices.index
        signal_data = pd.DataFrame(index=dates)

        # Term structure slope
        ts_slope = (self.strat_data.cm_vol_fut_prices[fd[1]]
                    - self.strat_data.cm_vol_fut_prices[fd[0]])
        if pct_ts:
            ts_slope /= self.strat_data.cm_vol_fut_prices[fd[0]]

        ann_factor = np.sqrt(constants.trading_days_per_year)
        rv = 100.0 * ann_factor * self.strat_data.index_fut_returns[
            self.settings.index_fut_ticker].ewm(com=21).std()

        x = pd.DataFrame(index=dates)
        x['front'] = self.strat_data.cm_vol_fut_prices[fd[0]]
        x['front_change'] = x['front'].diff(1).ewm(com=5).mean()
        x['rlzd_implied'] = rv - x['front']
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

    def initialize_term_structure_signals(self, **kwargs):

        fd = self.settings.fd

        # Looking at how long to smooth term structure
        windows = kwargs.get('windows', [0, 1, 5, 10])
        pct_ts = kwargs.get('pct_ts', True)

        dates = self.strat_data.cm_vol_fut_prices.index
        signal_data = pd.DataFrame(index=dates)

        ts_slope = (self.strat_data.cm_vol_fut_prices[fd[1]]
                    - self.strat_data.cm_vol_fut_prices[fd[0]])

        if pct_ts:
            ts_slope /= self.strat_data.cm_vol_fut_prices[fd[0]]

        for window in windows:
            signal_data['ts_' + str(window)] = ts_slope.ewm(com=window).mean()

        kwargs['signal_data'] = signal_data

        return signal_data

    def initialize_momentum_signals(self, **kwargs):

        # Studying impact of various time windows for prior performance
        windows = kwargs.get('windows', [5, 10, 21, 63])

        # Get the static backtest data just to be sure
        backtest_data = self.compute_static_strategy(**kwargs)

        dates = backtest_data.index
        signal_data = pd.DataFrame(index=dates)
        for window in windows:
            signal_data['mom_'+str(window)] = backtest_data['static_pnl'] \
                .ewm(com=window).mean()

        kwargs['signal_data'] = signal_data

        return signal_data

    def initialize_convexity_signals(self, **kwargs):

        fd = self.settings.fd
        sd = 0

        # Looking at how long to smooth term structure
        windows = kwargs.get('windows', [0, 1, 5, 10])
        pct_cv = kwargs.get('pct_cv', False)

        dates = self.strat_data.cm_vol_fut_prices.index
        signal_data = pd.DataFrame(index=dates)

        ts_slope_front = (self.strat_data.cm_vol_fut_prices[fd[0]]
                          - self.strat_data.cm_vol_fut_prices[sd])

        ts_slope_back = (self.strat_data.cm_vol_fut_prices[fd[1]]
                         - self.strat_data.cm_vol_fut_prices[fd[1] - 21])

        if pct_cv:
            ts_slope_front /= self.strat_data.cm_vol_fut_prices[sd]
            ts_slope_back /= self.strat_data.cm_vol_fut_prices[fd[1] - 21]

        convexity = ts_slope_front - ts_slope_back \
                                     / self.calc.rolling_front_back_beta

        for window in windows:
            signal_data['cv_' + str(window)] = convexity.ewm(com=window).mean()

        kwargs['signal_data'] = signal_data

        return signal_data

    def initialize_positioning_signals(self, **kwargs):

        dates = self.strat_data.cm_vol_fut_prices.index
        signal_data = pd.DataFrame(index=dates)
        com = kwargs.get('com', 252)

        signal_data['vol_spec_pos'] = self.strat_data.vol_spec_pos \
                                       / self.strat_data.vol_fut_open_int \
                                           .ewm(com=com).mean()

        # Note: these data don't come out until t + x?
        offset = 3
        signal_data = signal_data.shift(offset)

        # Fill to daily
        signal_data = signal_data.fillna(method='ffill')

        # Change signals
        signal_data['vol_spec_pos_chg_s'] = signal_data['vol_spec_pos'] \
            .diff(1).ewm(com=21).mean()

        signal_data['vol_spec_pos_chg_l'] = signal_data['vol_spec_pos'] \
            .diff(1).ewm(com=63).mean()

        kwargs['signal_data'] = signal_data

        return signal_data

    def initialize_signals(self, **kwargs):
        holding_period_days = kwargs.get('holding_period_days', 1)
        vol_target_com = kwargs.get('vol_target_com', 63)
        signals_z_cap = kwargs.get('signals_z_cap', 1.0)

        mom_signals = self.initialize_momentum_signals()
        ts_signals = self.initialize_term_structure_signals()
        pos_signals = self.initialize_positioning_signals()
        rts_signals = self.initialize_rel_ts_signals()
        cv_signals = self.initialize_convexity_signals()

        static_pnl = self.compute_static_strategy()

        mom_signal_pnl, mom_positions = self.backtest_signals(
            holding_period_days=holding_period_days,
            signal_data=mom_signals,
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        ts_signal_pnl, ts_positions = self.backtest_signals(
            holding_period_days=holding_period_days,
            signal_data=ts_signals,
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        cv_signal_pnl, cv_positions = self.backtest_signals(
            holding_period_days=holding_period_days,
            signal_data=cv_signals,
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        # Normalize direction
        normalize_direction = True
        if normalize_direction:
            mom_signal_dir = 1.0
            cv_signal_dir = -1.0
            ts_signal_dir = 1.0

            mom_signal_pnl *= mom_signal_dir
            cv_signal_pnl *= cv_signal_dir
            ts_signal_pnl *= ts_signal_dir

            pos_signals *= mom_signal_dir
            cv_signals *= cv_signal_dir
            ts_signals *= ts_signal_dir

        # Complete signal pnl dataset
        signal_pnl = pd.concat([mom_signal_pnl, ts_signal_pnl, cv_signal_pnl],
                               axis=1)
        signal_data = pd.concat([mom_signals, ts_signals, cv_signals], axis=1)
        kwargs['signal_data'] = signal_data
        signal_data_z = self.compute_signal_z(**kwargs)

        output = struct
        output.signal_data_z = signal_data_z
        output.signal_data = signal_data
        output.signal_pnl = signal_pnl
        output.static_pnl = static_pnl

        return output
