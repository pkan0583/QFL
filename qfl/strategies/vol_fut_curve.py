import pandas as pd
import numpy as np
import datetime as dt
import struct

import statsmodels.api as sm
import qfl.core.market_data as md
import qfl.core.constants as constants

from qfl.strategies.strategies import Strategy


class VixCurveStrategy(Strategy):

    @classmethod
    def initialize_data(cls, **kwargs):

        cls.settings.index_fut_series = kwargs.get('index_futures_series',
                                                   'ES')
        cls.settings.vol_fut_series = kwargs.get('vol_futures_series', 'VX')
        short_month = kwargs.get('short_month', 1)
        long_month = kwargs.get('long_month', 5)
        cls.settings.short_ticker = cls.settings.vol_fut_series \
                                    + str(short_month)
        cls.settings.long_ticker = cls.settings.vol_fut_series \
                                    + str(long_month)
        cls.settings.index_fut_ticker = cls.settings.index_fut_series + "1"

        # Beginning of semi-clean VIX futures data
        cls.settings.default_start_date = dt.datetime(2007, 3, 26)
        cls.settings.fd = [21 * short_month, 21 * long_month]

        cls.settings.start_date = kwargs.get('start_date',
                                             cls.settings.default_start_date)

        # Constant maturity prices
        cls.data.cm_vol_fut_prices = \
            md.get_constant_maturity_futures_prices_by_series(
                futures_series=cls.settings.vol_fut_series,
                start_date=cls.settings.start_date)['price'] \
                .unstack('days_to_maturity')\
                .reset_index(level='series', drop=True)

        # Rolling returns
        cls.data.vol_fut_returns = md.get_rolling_futures_returns_by_series(
            futures_series=cls.settings.vol_fut_series,
            start_date=cls.settings.start_date,
            level_change=True
        )

        # CM returns
        cls.data.cm_vol_fut_returns = cls.data.cm_vol_fut_prices.diff(1)

        # Positioning
        vol_spec_pos = \
            md.get_cftc_positioning_by_series(
                futures_series=cls.settings.vol_fut_series,
                start_date=cls.settings.start_date)
        vol_spec_pos = vol_spec_pos.set_index('date', drop=True)
        cls.data.vol_spec_pos = vol_spec_pos['lev_money_positions_long_all'] \
                                - vol_spec_pos['lev_money_positions_short_all']
        cls.data.vol_fut_open_int = vol_spec_pos['open_interest_all']

        # Index data for realized vol
        cls.data.index_fut_returns = md.get_rolling_futures_returns_by_series(
            futures_series=cls.settings.index_fut_series,
            start_date=cls.settings.start_date,
            level_change=False
        )

    @classmethod
    def compute_hedge_ratios(cls, **kwargs):

        rolling_beta_com = kwargs.get('rolling_beta_com', 63)

        # Predict long term volatility move as a function of short
        fd = cls.settings.fd
        y = cls.data.cm_vol_fut_returns[fd[1]]
        x = pd.DataFrame(index=cls.data.cm_vol_fut_returns.index)
        x['front_return'] = cls.data.cm_vol_fut_returns[fd[0]]
        x['front_level'] = cls.data.cm_vol_fut_prices[fd[0]].shift(1)
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
        cls.calc.front_back_beta = (r3.params.front_return
                                  + x['front_level']
                                  * r3.params.interaction)

        cls.calc.rolling_front_back_beta = \
            y.ewm(com=rolling_beta_com).cov(x['front_return']
                                            .ewm(com=rolling_beta_com)) \
            / x['front_return'].ewm(com=rolling_beta_com).var()

        buffer = 21
        cls.calc.rolling_front_back_beta.iloc[0:buffer] \
            = cls.calc.rolling_front_back_beta.iloc[buffer]

        cls.calc.r1 = r1
        cls.calc.r2 = r2
        cls.calc.r3 = r3

    @classmethod
    def compute_static_strategy(cls, **kwargs):

        fd = cls.settings.fd
        st = cls.settings.short_ticker
        lt = cls.settings.long_ticker

        # Vega-vs-delta (short vol, short stocks)
        cls.calc.backtest_data \
            = pd.DataFrame(index=cls.data.curve_returns.index)

        cls.calc.backtest_data['front'] = cls.data.vol_fut_returns[st]
        cls.calc.backtest_data['back'] = cls.data.vol_fut_returns[lt] \
            / cls.calc.rolling_front_back_beta

        cls.calc.backtest_data['static_pnl'] \
            = cls.calc.backtest_data['front'] - cls.calc.backtest_data['back']

        return cls.calc.backtest_data

    @classmethod
    def initialize_rel_ts_signals(cls, **kwargs):

        # Idea here is, how steep is the curve conditional on its level?

        fd = cls.settings.fd
        sd = 0

        windows = kwargs.get('windows', [1, 5, 10])
        expanding = kwargs.get('expanding', True)
        reg_window = kwargs.get('reg_window', 512)
        pct_ts = kwargs.get('pct_ts', False)

        dates = cls.data.cm_vol_fut_prices.index
        signals_data = pd.DataFrame(index=dates)

        # Term structure slope
        ts_slope = (cls.data.cm_vol_fut_prices[fd[1]]
                    - cls.data.cm_vol_fut_prices[fd[0]])
        if pct_ts:
            ts_slope /= cls.data.cm_vol_fut_prices[fd[0]]

        ann_factor = np.sqrt(constants.trading_days_per_year)
        rv = 100.0 * ann_factor * cls.data.index_fut_returns[
            cls.settings.index_fut_ticker].ewm(com=21).std()

        x = pd.DataFrame(index=dates)
        x['front'] = cls.data.cm_vol_fut_prices[fd[0]]
        x['front_change'] = x['front'].diff(1).ewm(com=5).mean()
        x['rlzd_implied'] = rv - x['front']
        x = x[np.isfinite(x).all(axis=1)]

        if expanding:
            rolling_reg = pd.ols(y=ts_slope, x=x, window_type='expanding')
        else:
            rolling_reg = pd.ols(y=ts_slope, x=x, window=reg_window)

        for window in windows:
            signals_data['c_'+str(window)] = rolling_reg\
                                            .resid\
                                            .ewm(com=window)\
                                            .mean()

        kwargs['signals_data'] = signals_data
        return signals_data

    @classmethod
    def initialize_term_structure_signals(cls, **kwargs):

        fd = cls.settings.fd

        # Looking at how long to smooth term structure
        windows = kwargs.get('windows', [0, 1, 5, 10])
        pct_ts = kwargs.get('pct_ts', True)

        dates = cls.data.cm_vol_fut_prices.index
        signals_data = pd.DataFrame(index=dates)

        ts_slope = (cls.data.cm_vol_fut_prices[fd[1]]
                  - cls.data.cm_vol_fut_prices[fd[0]])

        if pct_ts:
            ts_slope /= cls.data.cm_vol_fut_prices[fd[0]]

        for window in windows:
            signals_data['ts_' + str(window)] = ts_slope.ewm(com=window).mean()

        kwargs['signals_data'] = signals_data

        return signals_data

    @classmethod
    def initialize_momentum_signals(cls, **kwargs):

        # Studying impact of various time windows for prior performance
        windows = kwargs.get('windows', [5, 10, 21, 63])

        # Get the static backtest data just to be sure
        backtest_data = cls.compute_static_strategy(**kwargs)

        dates = backtest_data.index
        signals_data = pd.DataFrame(index=dates)
        for window in windows:
            signals_data['mom_'+str(window)] = backtest_data['static_pnl'] \
                .ewm(com=window).mean()

        kwargs['signals_data'] = signals_data

        return signals_data

    @classmethod
    def initialize_convexity_signals(cls, **kwargs):

        fd = cls.settings.fd
        sd = 0

        # Looking at how long to smooth term structure
        windows = kwargs.get('windows', [0, 1, 5, 10])
        pct_cv = kwargs.get('pct_cv', False)

        dates = cls.data.cm_vol_fut_prices.index
        signals_data = pd.DataFrame(index=dates)

        ts_slope_front = (cls.data.cm_vol_fut_prices[fd[0]]
                         - cls.data.cm_vol_fut_prices[sd])

        ts_slope_back = (cls.data.cm_vol_fut_prices[fd[1]]
                          - cls.data.cm_vol_fut_prices[fd[1]-21])

        if pct_cv:
            ts_slope_front /= cls.data.cm_vol_fut_prices[sd]
            ts_slope_back /= cls.data.cm_vol_fut_prices[fd[1]-21]

        convexity = ts_slope_front - ts_slope_back \
                                     / cls.calc.rolling_front_back_beta

        for window in windows:
            signals_data['cv_' + str(window)] = convexity.ewm(com=window).mean()

        kwargs['signals_data'] = signals_data

        return signals_data

    @classmethod
    def initialize_positioning_signals(cls, **kwargs):

        dates = cls.data.cm_vol_fut_prices.index
        signals_data = pd.DataFrame(index=dates)
        com = kwargs.get('com', 252)

        signals_data['vol_spec_pos'] = cls.data.vol_spec_pos \
                                       / cls.data.vol_fut_open_int \
                                           .ewm(com=com).mean()

        # Note: these data don't come out until t + x?
        offset = 3
        signals_data = signals_data.shift(offset)

        # Fill to daily
        signals_data = signals_data.fillna(method='ffill')

        # Change signals
        signals_data['vol_spec_pos_chg_s'] = signals_data['vol_spec_pos'] \
            .diff(1).ewm(com=21).mean()

        signals_data['vol_spec_pos_chg_l'] = signals_data['vol_spec_pos'] \
            .diff(1).ewm(com=63).mean()

        kwargs['signals_data'] = signals_data

        return signals_data

    @classmethod
    def initialize_signals(cls, **kwargs):
        holding_period_days = kwargs.get('holding_period_days', 1)
        vol_target_com = kwargs.get('vol_target_com', 63)
        signals_z_cap = kwargs.get('signals_z_cap', 1.0)

        mom_signals = cls.initialize_momentum_signals()
        ts_signals = cls.initialize_term_structure_signals()
        pos_signals = cls.initialize_positioning_signals()
        rts_signals = cls.initialize_rel_ts_signals()
        cv_signals = cls.initialize_convexity_signals()

        static_pnl = cls.compute_static_strategy()

        mom_signal_pnl, mom_positions = cls.backtest_signals(
            holding_period_days=holding_period_days,
            signals_data=mom_signals,
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        ts_signal_pnl, ts_positions = cls.backtest_signals(
            holding_period_days=holding_period_days,
            signals_data=ts_signals,
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        cv_signal_pnl, cv_positions = cls.backtest_signals(
            holding_period_days=holding_period_days,
            signals_data=cv_signals,
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
        signals_data = pd.concat([mom_signals, ts_signals, cv_signals], axis=1)
        kwargs['signals_data'] = signals_data
        signals_data_z = cls.compute_signal_z(**kwargs)

        output = struct
        output.signals_data_z = signals_data_z
        output.signals_data = signals_data
        output.signal_pnl = signal_pnl
        output.static_pnl = static_pnl

        return output
