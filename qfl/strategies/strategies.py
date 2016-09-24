import pandas as pd
import numpy as np
import datetime as dt
import struct

import statsmodels.api as sm
import qfl.core.market_data as md
import qfl.core.constants as constants
from scipy.optimize import minimize


class Strategy(object):

    data = struct
    calc = struct
    settings = struct

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
    def backtest_signals(cls, **kwargs):

        backtest_data = cls.compute_static_strategy(**kwargs)
        dates = backtest_data.index

        holding_period_days = kwargs.get('holding_period_days', 1)
        vol_target_com = kwargs.get('vol_target_com', None)

        signals_data_z = cls.compute_signal_z(**kwargs)

        backtest_signals = signals_data_z.columns
        signal_pnl = pd.DataFrame(index=dates,
                                  columns=backtest_signals)

        positions = signals_data_z.shift(holding_period_days)

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
        er_shrinkage = kwargs.get('er_shrinkage', 0.25)

        # Prepare model
        signal_corr_adj, signal_cov_adj, signal_er_cov = \
            PortfolioOptimizer.prepare_uncertainty_model(
                er=signal_er,
                cov_r=signal_cov,
                corr_r=signal_corr,
                corr_r_shrinkage=signal_corr_shrinkage,
                corr_er_shrinkage=signal_er_corr_shrinkage,
                er_se_beta_to_er=signal_se_beta_to_er,
                er_se_beta_to_vol=signal_se_beta_to_vol,
                er_shrinkage=er_shrinkage
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


class PortfolioStrategy(Strategy):

    """
    This guy represents a Strategy where there are lots of underliers and
    you form a portfolio by buying or selling based on signals, rather than
    a standard strategy where you have a single static trade and then
    buy or sell it based on signals
    """

    _universe = None

    @classmethod
    def set_universe(cls, **kwargs):
        # This has to be defined by the specific implementation
        raise NotImplementedError

    @classmethod
    def get_universe(cls, **kwargs):
        return cls._universe

    @classmethod
    def backtest_signals(cls, **kwargs):
        x=1


    @classmethod
    def compute_signal_quantile_performance(cls, **kwargs):
        """
        This is a highly idealized notion of signal performance involving
        mid-market daily trading... arguably very subject to data noise
        though actually surprisingly comparable to the proper version
        :param kwargs:
        :return:
        """

        # Methods: 'top_bottom_n', 'top_bottom_pct'
        method = kwargs.get('method', 'top_bottom_q')
        signal_data = kwargs.get('signal_data', None)

        quantiles = kwargs.get('quantiles', [0.0, 0.20, 0.40, 0.60, 0.80, 1.0])
        if quantiles[0] > 0:
            quantiles = [0] + quantiles

        # This should be the core data which has "tot_pnl" as a field
        data = kwargs.get('data', None)

        signal_pctile = pd.DataFrame(index=signal_data.index,
                                     columns=signal_data.columns)

        signal_pctile = signal_pctile.unstack('ticker')

        for sig in signal_data.columns:
            signal_data_sig = signal_data[sig].unstack('ticker')

            # Cross-sectional percentile
            signal_pctile[sig] = signal_data_sig.rank(axis=1, pct=True)

        signal_positions = dict()
        signal_pnl = dict()
        data = data.unstack('ticker')

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

                signal_pnl[q][sig] = signal_positions[q][sig] * data['tot_pnl']
                signal_pnl[q][sig] = signal_pnl[q][sig].fillna(value=0)

            # Replace NONE with NAN
            signal_positions[q] = signal_positions[q].where(
                (pd.notnull(signal_positions[q])), 0)

        return signal_pnl, signal_positions, signal_pctile


class SignalAnalyzer(object):

    @staticmethod
    def compute_time_varying_expected_return():

        # A Bayesian algo would do something along the lines of a rolling
        # return with capping of outliers based on previous rolling return

        x=1


class PortfolioOptimizer(object):

    @staticmethod
    def prepare_uncertainty_model(er=None,
                                  cov_r=None,
                                  corr_r=None,
                                  corr_r_shrinkage=0.90,
                                  corr_er_shrinkage=0.70,
                                  er_se_beta_to_er=0.25,
                                  er_se_beta_to_vol=0.75,
                                  er_shrinkage=0.25):

        # Expected return shrink towards zero
        er = (1 - er_shrinkage) * er

        # Signal correlation and covariance shrinkage
        for row in cov_r.columns:
            for col in cov_r.columns:
                if row != col:
                    corr_r.loc[row, col] *= corr_r_shrinkage
                    cov_r.loc[row, col] *= corr_r_shrinkage

        # Forward-looking expected return correlation
        signal_er_corr = np.sign(corr_r) * np.abs(corr_r) \
                         * corr_er_shrinkage
        for col in signal_er_corr:
            signal_er_corr.loc[col, col] = 1.0

        # Forward-looking expected return covariance
        signal_er_cov = pd.DataFrame(index=signal_er_corr.index,
                                     columns=signal_er_corr.columns)
        for i in signal_er_corr.index:
            for j in signal_er_corr.index:
                signal_er_cov.loc[i, j] = er.loc[i] * er.loc[j] \
                                          * er_se_beta_to_er * \
                                          signal_er_corr.loc[i, j]
                signal_er_cov.loc[i, j] += cov_r.loc[i, j] * er_se_beta_to_vol

        return corr_r, cov_r, signal_er_cov

    @staticmethod
    def compute_portfolio_weights(weights_guess=None,
                                  er=None,
                                  cov_r=None,
                                  cov_er=None,
                                  time_horizon_days=None,
                                  sum_weights_constraint=1.0,
                                  long_only=False):

        # Default = equal weight guess
        if weights_guess is None:
            weights_guess = np.ones(len(er) - 1) / (len(er))

        # Long-only constraint implemented via long/exp transforms
        if long_only:
            weights_guess = np.log(weights_guess)

        args = (er,
                cov_r,
                cov_er,
                time_horizon_days,
                long_only,
                sum_weights_constraint)

        optim = minimize(PortfolioOptimizer._compute_negative_ir,
                         weights_guess, args=args, options={'disp': True})

        if long_only:
            weights = np.append(np.exp(optim.x), 1 - np.sum(np.exp(optim.x)))
            if weights[-1] < 0:
                weights[-1] = 0
        else:
            weights = np.append(optim.x, 1 - np.sum(optim.x))

        portfolio_ir_adjusted, portfolio_ir_traditional, p_var_r, p_var_er \
            = PortfolioOptimizer.compute_ir(
                weights, er, cov_r, cov_er, time_horizon_days)

        return weights, portfolio_ir_adjusted, optim

    @staticmethod
    def compute_ir(weights=None,
                   er=None,
                   cov_r=None,
                   cov_er=None,
                   time_horizon_days=None):

        cov_r_daily = cov_r / constants.trading_days_per_year
        cov_er_daily = cov_er / constants.trading_days_per_year ** 2.0

        portfolio_er = weights.transpose().dot(er) \
                       * time_horizon_days / constants.trading_days_per_year

        p_var_r = weights.transpose().dot(cov_r_daily).dot(weights) \
                  * time_horizon_days
        p_var_er = weights.transpose().dot(cov_er_daily).dot(weights) \
                   * time_horizon_days ** 2.0

        portfolio_ir_adjusted = portfolio_er / np.sqrt(p_var_r + p_var_er)
        portfolio_ir_traditional = portfolio_er / np.sqrt(p_var_r)

        return portfolio_ir_adjusted,\
               portfolio_ir_traditional,\
               p_var_r,\
               p_var_er

    @staticmethod
    def _compute_negative_ir(weights=None,
                             er=None,
                             cov_r=None,
                             cov_er=None,
                             time_horizon_days=None,
                             long_only=False,
                             sum_weights_constraint=1.0):

        # We implement the long-only constraint via log/exp transformations
        if long_only:
            weights = np.exp(weights)

        # Optimization case: constrained, don't set the last weight
        if len(weights) == len(er) - 1:
            last_weight = sum_weights_constraint - np.sum(weights)
            weights = np.append(weights, np.array([last_weight]))

        portfolio_ir_adjusted, portfolio_ir_traditional, p_var_r, p_var_er \
            = PortfolioOptimizer.compute_ir(weights=weights,
                                            er=er,
                                            cov_r=cov_r,
                                            cov_er=cov_er,
                                            time_horizon_days=time_horizon_days)
        return -portfolio_ir_adjusted

    @staticmethod
    def compute_signal_portfolio_sensitivity(strategy=None,
                                             signals_data=None,
                                             weights=None,
                                             num_sims=1000,
                                             sigma=2.0,
                                             signals_z_cap=1.0,
                                             holding_period_days=1,
                                             vol_target_com=63):

        perf_pctiles = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]

        randoms = np.random.randn(num_sims, len(weights)) * sigma / len(weights)
        randoms = pd.DataFrame(data=randoms, columns=signals_data.columns)

        sim_perf = pd.DataFrame(index=signals_data.index,
                                columns=range(0, num_sims))

        signal_z = pd.DataFrame(index=signals_data.index,
                                columns=signals_data.columns)
        for signal in weights.index:
            signal_z[signal] = strategy.compute_signal_z(
                signals_data=signals_data[signal],
                signals_z_cap=signals_z_cap)

        for s in range(0, num_sims):

            if np.mod(s, 100) == 0:
                print('simulation ' + str(s) + " out of " + str(num_sims))

            sim_weights = weights['weight'] + randoms.iloc[s]
            sim_weights[sim_weights < 0.0] = 0.0
            sim_weights /= sim_weights.sum()

            sim_signal = pd.DataFrame(index=signals_data.index,
                                      columns=['x'])
            sim_signal['x'] = 0.0

            for signal in weights.index:
                sim_signal['x'] += sim_weights.loc[signal] * signal_z[signal]

            # Combined pnl
            sim_perf[s], pos = strategy.backtest_signals(
                holding_period_days=holding_period_days,
                signals_data=pd.DataFrame(sim_signal['x']),
                vol_target_com=vol_target_com,
                signals_z_cap=signals_z_cap)

        sim_perf_total = sim_perf.sum(axis=0).sort_values()
        sim_perf = sim_perf[sim_perf_total.index]

        sim_perf_percentiles = pd.DataFrame(index=signals_data.index,
                                            columns=perf_pctiles)
        for pctile in perf_pctiles:
            sim_perf_percentiles[pctile] = \
                sim_perf[sim_perf.columns[np.floor(num_sims * pctile)]]

        return sim_perf_percentiles, sim_perf

    @staticmethod
    def compute_rv_signal_portfolio_sensitivity(signals_pnl=None,
                                                weights=None,
                                                num_sims=1000,
                                                sigma=2.0):
        perf_pctiles = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]

        randoms = np.random.randn(num_sims, len(weights)) * sigma / len(weights)
        randoms = pd.DataFrame(data=randoms, columns=signals_pnl.columns)

        sim_perf = pd.DataFrame(index=signals_pnl.index,
                                columns=range(0, num_sims))

        for s in range(0, num_sims):

            if np.mod(s, 100) == 0:
                print('simulation ' + str(s) + " out of " + str(num_sims))

            sim_weights = weights['weight'] + randoms.iloc[s]
            sim_weights[sim_weights < 0.0] = 0.0
            sim_weights /= sim_weights.sum()

            sim_perf[s] = 0
            for sig in signals_pnl.columns:
                sim_perf[s] += signals_pnl[sig] * sim_weights[sig]

        sim_perf_total = sim_perf.sum(axis=0).sort_values()
        sim_perf = sim_perf[sim_perf_total.index]

        sim_perf_percentiles = pd.DataFrame(index=signals_pnl.index,
                                            columns=perf_pctiles)
        for pctile in perf_pctiles:
            sim_perf_percentiles[pctile] = \
                sim_perf[sim_perf.columns[np.floor(num_sims * pctile)]]

        return sim_perf_percentiles, sim_perf

