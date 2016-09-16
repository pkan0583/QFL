import pandas as pd
import numpy as np
import datetime as dt
import struct

import statsmodels.api as sm
import qfl.core.market_data as md
import qfl.core.constants as constants

from qfl.strategies.strategies import Strategy, PortfolioOptimizer


class VegaVsDeltaStrategy(Strategy):

    @classmethod
    def initialize_data(cls, **kwargs):

        # Beginning of semi-clean VIX futures data
        cls.settings.default_start_date = dt.datetime(2007, 3, 26)
        cls.settings.fd = 21

        cls.settings.start_date = kwargs.get('start_date',
                                             cls.settings.default_start_date)
        cls.settings.vol_fut_series = kwargs.get('vol_futures_series',
                                                 'VX')
        cls.settings.index_fut_series = kwargs.get('index_futures_series',
                                                   'ES')
        cls.settings.vol_fut_ticker = cls.settings.vol_fut_series + '1'
        cls.settings.index_fut_ticker = cls.settings.index_fut_series + '1'

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

        cls.data.index_fut_returns = md.get_rolling_futures_returns_by_series(
            futures_series=cls.settings.index_fut_series,
            start_date=cls.settings.start_date,
            level_change=False
        )

        # Positioning

        vol_spec_pos = \
            md.get_cftc_positioning_by_series(
                futures_series=cls.settings.vol_fut_series,
                start_date=cls.settings.start_date)
        vol_spec_pos = vol_spec_pos.set_index('date', drop=True)
        cls.data.vol_spec_pos = vol_spec_pos['lev_money_positions_long_all'] \
                              - vol_spec_pos['lev_money_positions_short_all']
        cls.data.vol_fut_open_int = vol_spec_pos['open_interest_all']

        index_spec_pos = \
            md.get_cftc_positioning_by_series(
                futures_series=cls.settings.index_fut_series,
                start_date=cls.settings.start_date)
        index_spec_pos = index_spec_pos.set_index('date', drop=True)
        cls.data.index_spec_pos = index_spec_pos['lev_money_positions_long_all']\
                                - index_spec_pos['lev_money_positions_short_all']
        cls.data.index_fut_open_int = index_spec_pos['open_interest_all']

    @classmethod
    def initialize_signals(cls, **kwargs):

        holding_period_days = kwargs.get('holding_period_days', 1)
        vol_target_com = kwargs.get('vol_target_com', 63)
        signals_z_cap = kwargs.get('signals_z_cap', 1.0)

        mom_signals = cls.initialize_momentum_signals()
        ts_signals = cls.initialize_term_structure_signals()
        pos_signals = cls.initialize_positioning_signals()
        rv_signals = cls.initialize_rlzd_vs_implied_signals()
        svb_signals = cls.initialize_spot_vol_beta_signals()
        c_signals = cls.initialize_complex_signals()

        static_pnl = cls.compute_static_strategy()

        pos_signal_pnl, pos_positions = cls.backtest_signals(
            holding_period_days=holding_period_days,
            signals_data=pos_signals,
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        ts_signal_pnl, ts_positions = cls.backtest_signals(
            holding_period_days=holding_period_days,
            signals_data=ts_signals,
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        rv_signal_pnl, rv_positions = cls.backtest_signals(
            holding_period_days=holding_period_days,
            signals_data=rv_signals,
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
        signals_data = pd.concat([pos_signals, ts_signals, rv_signals], axis=1)
        kwargs['signals_data'] = signals_data
        signals_data_z = cls.compute_signal_z(**kwargs)

        output = struct
        output.signals_data_z = signals_data_z
        output.signals_data = signals_data
        output.signal_pnl = signal_pnl
        output.static_pnl = static_pnl

        return output

    @classmethod
    def compute_signal_portfolio_optimization(cls,
                                              signal_er=None,
                                              signal_cov=None,
                                              signal_corr=None,
                                              **kwargs):

        # Parameters for portfolio construction exercise
        time_horizon = kwargs.get('time_horizon', 5 * 252)
        signal_corr_shrinkage = kwargs.get('signal_corr_shrinkage', 0.90)
        signal_se_beta_to_er = kwargs.get('signal_se_beta_to_er', 0.25)
        signal_se_beta_to_vol = kwargs.get('signal_se_beta_to_vol', 0.75)
        signal_er_corr_shrinkage = kwargs.get('signal_er_corr_shrinkage', 0.80)
        sum_of_weights = kwargs.get('sum_of_weights', 1.0)

        # Prepare model
        signal_corr_adj, signal_cov_adj, signal_er_cov = \
            PortfolioOptimizer.prepare_uncertainty_model(
                er=signal_er,
                cov_r=signal_cov,
                corr_r=signal_corr,
                corr_r_shrinkage=signal_corr_shrinkage,
                corr_er_shrinkage=signal_er_corr_shrinkage,
                er_se_beta_to_er=signal_se_beta_to_er,
                er_se_beta_to_vol=signal_se_beta_to_vol
            )

        # Signal portfolio optimization
        weights, portfolio_ir_adjusted, optim = \
            PortfolioOptimizer.compute_portfolio_weights(
                er=signal_er,
                cov_r=signal_cov,
                cov_er=signal_er_cov,
                time_horizon_days=time_horizon,
                sum_weights_constraint=sum_of_weights,
                long_only=True)

        # Information ratio handicapping
        port_er_sd = weights.transpose().dot(signal_er_cov).dot(weights) ** 0.5

        # Output formatting
        weights = pd.DataFrame(data=weights, index=signal_er.index,
                               columns=['weight'])

        output = struct
        output.weights = weights
        output.port_er_sd = port_er_sd
        output.portfolio_ir_adjusted = portfolio_ir_adjusted
        output.optim = optim

        return output

    @classmethod
    def compute_master_backtest(cls, **kwargs):

        # Basic parameters
        holding_period_days = kwargs.get('holding_period_days', 1)
        vol_target_com = kwargs.get('vol_target_com', 63)
        signals_z_cap = kwargs.get('signals_z_cap', 1.0)
        rolling_beta_com = kwargs.get('signals_z_cap', 126)
        transaction_cost_per_unit = kwargs.get('transaction_cost_per_unit', 0.05)

        # Hedge ratios
        cls.compute_hedge_ratios(rolling_com=rolling_beta_com)

        # Signals data and PNL
        signal_output = cls.initialize_signals(**kwargs)

        signal_er = signal_output.signal_pnl.mean() \
                    * constants.trading_days_per_year

        # Single in-sample covariance matrix
        signal_cov = signal_output.signal_pnl.cov()\
                     * constants.trading_days_per_year
        signal_corr = signal_output.signal_pnl.corr()

        # Optimization
        optim_output = cls.compute_signal_portfolio_optimization(
            signal_er=signal_er,
            signal_corr=signal_corr,
            signal_cov=signal_cov,
            **kwargs)

        # Combined signal
        combined_signal = pd.DataFrame(index=signal_output.signal_pnl.index,
                                       columns=['optim_weight', 'equal_weight'])
        combined_signal[['optim_weight', 'equal_weight']] = 0.0

        for signal in optim_output.weights.index:
            sz = cls.compute_signal_z(
                signals_data=signal_output.signals_data[signal],
                signals_z_cap=signals_z_cap)
            combined_signal['optim_weight'] += optim_output.weights \
                .loc[signal].values[0] * sz
            combined_signal['equal_weight'] += \
                sz * (1.0 / len(optim_output.weights.index))

        # Combined pnl
        combined_pnl = pd.DataFrame(index=signal_output.signal_pnl.index,
                                    columns=['optim_weight', 'equal_weight'])

        combined_pnl['optim_weight'], optim_positions = cls.backtest_signals(
            holding_period_days=holding_period_days,
            signals_data=pd.DataFrame(combined_signal['optim_weight']),
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        combined_pnl['equal_weight'], ew_positions = cls.backtest_signals(
            holding_period_days=holding_period_days,
            signals_data=pd.DataFrame(combined_signal['equal_weight']),
            vol_target_com=vol_target_com,
            signals_z_cap=signals_z_cap)

        transactions = np.abs(combined_signal.diff(1))

        combined_pnl_net = combined_pnl - transactions \
                                          * transaction_cost_per_unit

        combined_pnl = combined_pnl[np.isfinite(
            pd.to_numeric(combined_pnl['optim_weight']))]

        output = struct
        output.combined_pnl_net = combined_pnl_net
        output.combined_pnl = combined_pnl
        output.positions = optim_positions
        output.optim_output = optim_output
        output.signal_output = signal_output

        return output

    @classmethod
    def compute_signal_z(cls, **kwargs):

        signals_data = kwargs.get('signals_data', None)
        expanding = kwargs.get('expanding', True)
        signals_z_cap = kwargs.get('signals_z_cap', 2.0)
        if not expanding:
            com = kwargs.get('window', 252)

        if expanding:
            signals_data_z = (signals_data
                            - signals_data.expanding().mean()) \
                            / signals_data.expanding().std()
        else:
            signals_data_z = (signals_data
                             - signals_data.ewm(com=com).mean()) \
                            / signals_data.ewm(com=com).std()

        signals_data_z = signals_data_z.clip(lower=-signals_z_cap,
                                             upper=signals_z_cap)

        return signals_data_z

    @classmethod
    def initialize_rlzd_vs_implied_signals(cls, **kwargs):

        fd = cls.settings.fd

        dates = cls.data.index_fut_returns.index
        signals_data = pd.DataFrame(index=dates)

        windows = kwargs.get('windows', [10, 21, 42, 63])
        ann_factor = np.sqrt(constants.trading_days_per_year)

        for window in windows:
            signals_data['rv_'+str(window)] = 100.0 * ann_factor * \
                cls.data.index_fut_returns[
                cls.settings.index_fut_ticker].ewm(com=window).std()\
                - cls.data.cm_vol_fut_prices[fd]

        kwargs['signals_data'] = signals_data

        return signals_data

    @classmethod
    def initialize_complex_signals(cls, **kwargs):

        fd = cls.settings.fd
        sd = 0

        windows = kwargs.get('windows', [1, 5, 10])
        expanding = kwargs.get('expanding', True)
        reg_window = kwargs.get('reg_window', 512)

        dates = cls.data.index_fut_returns.index
        signals_data = pd.DataFrame(index=dates)

        # Term structure slope
        ts_slope = (cls.data.cm_vol_fut_prices[fd] - cls.data.cm_vol_fut_prices[
            sd]) / cls.data.cm_vol_fut_prices[fd]

        ann_factor = np.sqrt(constants.trading_days_per_year)
        rv = 100.0 * ann_factor * cls.data.index_fut_returns[
            cls.settings.index_fut_ticker].ewm(com=21).std()

        x = pd.DataFrame(index=dates)
        x['ivol'] = cls.data.cm_vol_fut_prices[fd]
        x['ivol_change'] = cls.data.cm_vol_fut_prices[fd].diff(
            1).ewm(com=5).mean()
        x['returns'] = cls.data.index_fut_returns[
            cls.settings.index_fut_ticker].ewm(com=5).mean()
        x['rlzd_implied'] = rv - x['ivol']
        x = x[np.isfinite(x).all(axis=1)]

        if expanding:
            rolling_reg = pd.ols(y=ts_slope, x=x, window_type='expanding')
        else:
            rolling_reg = pd.ols(y=ts_slope, x=x, window=reg_window)

        for window in windows:
            signals_data['c_'+str(window)] = rolling_reg.resid.ewm(com=window).mean()

        kwargs['signals_data'] = signals_data

        return signals_data

    @classmethod
    def initialize_spot_vol_beta_signals(cls, **kwargs):

        windows = kwargs.get('windows', [10, 21, 42, 63])

        dates = cls.data.index_fut_returns.index
        signals_data = pd.DataFrame(index=dates)

        y = cls.data.cm_vol_fut_prices[cls.settings.fd].diff(1)
        x = cls.data.index_fut_returns[cls.settings.index_fut_ticker]

        for window in windows:
            signals_data['svb_'+str(window)] = \
                y.ewm(com=window).cov(x.ewm(com=window)) \
                / x.ewm(com=window).var()

        kwargs['signals_data'] = signals_data

        return signals_data

    @classmethod
    def initialize_positioning_signals(cls, **kwargs):

        dates = cls.data.index_fut_returns.index
        signals_data = pd.DataFrame(index=dates)
        com = kwargs.get('com', 252)

        signals_data['vol_spec_pos'] = cls.data.vol_spec_pos \
                                     / cls.data.vol_fut_open_int\
                                     .ewm(com=com).mean()
        signals_data['index_spec_pos'] = cls.data.index_spec_pos\
                                       / cls.data.index_fut_open_int\
                                       .ewm(com=com).mean()

        # Note: these data don't come out until t + x?
        offset = 3
        signals_data = signals_data.shift(offset)

        # Fill to daily
        signals_data = signals_data.fillna(method='ffill')

        # Change signals
        signals_data['vol_spec_pos_chg_s'] = signals_data['vol_spec_pos']\
            .diff(1).ewm(com=21).mean()
        signals_data['index_spec_pos_chg_s'] = signals_data['index_spec_pos'] \
            .diff(1).ewm(com=21).mean()

        signals_data['vol_spec_pos_chg_l'] = signals_data['vol_spec_pos']\
            .diff(1).ewm(com=63).mean()
        signals_data['index_spec_pos_chg_l'] = signals_data['index_spec_pos'] \
            .diff(1).ewm(com=63).mean()

        kwargs['signals_data'] = signals_data

        return signals_data

    @classmethod
    def initialize_term_structure_signals(cls, **kwargs):

        fd = cls.settings.fd
        sd = 0

        # Looking at how long to smooth term structure
        windows = kwargs.get('windows', [0, 1, 5, 10])

        dates = cls.data.index_fut_returns.index
        signals_data = pd.DataFrame(index=dates)

        ts_slope = (cls.data.cm_vol_fut_prices[fd]
            - cls.data.cm_vol_fut_prices[sd]) / \
            cls.data.cm_vol_fut_prices[fd]

        for window in windows:
            signals_data['ts_'+str(window)] = ts_slope.ewm(com=window).mean()

        kwargs['signals_data'] = signals_data

        return signals_data

    @classmethod
    def initialize_momentum_signals(cls, **kwargs):

        # Studying impact of various time windows for prior performance
        windows = kwargs.get('windows', [1, 5, 10, 21])

        # Get the static backtest data just to be sure
        backtest_data = cls.compute_static_strategy(**kwargs)

        dates = cls.data.index_fut_returns.index
        signals_data = pd.DataFrame(index=dates)
        for window in windows:
            signals_data['mom_'+str(window)] = backtest_data['static_pnl'] \
                .ewm(com=window).mean()

        kwargs['signals_data'] = signals_data

        return signals_data

    @classmethod
    def compute_hedge_ratios(cls, **kwargs):

        fd = cls.settings.fd
        rolling_beta_com = kwargs.get('rolling_beta_com', 21)

        # Trailing betas
        y = cls.data.cm_vol_fut_prices[fd].diff(1)

        # Predict volatility move as a function of stuff
        x = pd.DataFrame(index=cls.data.index_fut_returns.index)
        x['index_return'] = cls.data.index_fut_returns[
            cls.settings.index_fut_ticker]
        x['vol_level'] = cls.data.cm_vol_fut_prices[fd].shift(1)
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
        cls.calc.spot_vol_beta = (r3.params.index_return
                                  + x['vol_level']
                                  * r3.params.interaction) / 100.0

        cls.calc.rolling_spot_vol_beta = \
            y.ewm(com=rolling_beta_com).cov(x['index_return']
                .ewm(com=rolling_beta_com)) \
            / x['index_return'].ewm(com=rolling_beta_com).var()

    @classmethod
    def compute_static_strategy(cls, **kwargs):

        # Vega-vs-delta (short vol, short stocks)
        cls.calc.backtest_data \
            = pd.DataFrame(index=cls.data.index_fut_returns.index)

        cls.calc.backtest_data['vega'] \
            = cls.data.vol_fut_returns[cls.settings.vol_fut_ticker]

        cls.calc.backtest_data['delta'] \
            = cls.data.index_fut_returns[cls.settings.index_fut_ticker]\
            * 100.0 * cls.calc.spot_vol_beta

        cls.calc.backtest_data['static_pnl'] \
            = cls.calc.backtest_data['delta']\
            - cls.calc.backtest_data['vega']

        return cls.calc.backtest_data

    @classmethod
    def backtest_signals(cls, **kwargs):

        dates = cls.data.index_fut_returns.index
        holding_period_days = kwargs.get('holding_period_days', 1)
        vol_target_com = kwargs.get('vol_target_com', None)

        signals_data_z = cls.compute_signal_z(**kwargs)

        backtest_signals = signals_data_z.columns
        signal_pnl = pd.DataFrame(index=dates,
                                  columns=backtest_signals)

        positions = signals_data_z.shift(holding_period_days)
        backtest_data = cls.compute_static_strategy(**kwargs)

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
