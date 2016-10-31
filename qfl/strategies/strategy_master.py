
import datetime as dt
import numpy as np
import pandas as pd
import struct

import qfl.core.calcs as calcs
import qfl.core.market_data as md
import qfl.core.constants as constants
import qfl.macro.macro_models as macro
import qfl.utilities.basic_utilities as utils
import qfl.core.portfolio_utils as putils

from qfl.macro.macro_models import FundamentalMacroModel
from qfl.core.database_interface import DatabaseInterface as db
from qfl.utilities.chart_utilities import format_plot
from qfl.utilities.nlp import DocumentsAnalyzer as da
import qfl.utilities.statistics as stats
from qfl.core.market_data import VolatilitySurfaceManager

import qfl.models.volatility_factor_model as vfm
from qfl.utilities.statistics import RollingFactorModel
from qfl.models.volatility_factor_model import VolatilityFactorModel

import qfl.strategies.strategies as strat
from qfl.strategies.vix_curve import VixCurveStrategy
from qfl.strategies.equity_vs_vol import VegaVsDeltaStrategy
from qfl.strategies.equity_vs_vol import VegaVsDeltaStrategy
from qfl.strategies.vol_rv import VolswapRvStrategy
from qfl.strategies.strategies import PortfolioOptimizer
import qfl.utilities.statistics as stats

import logging
logger = logging.getLogger()


class StrategyMaster(object):

    # Vision here:
    # - each strategy is a Strategy type
    # - each strategy has <initialize, run, output>
    # - each strategy has a return stream, equal/optim/for each signal

    strategy_names = ['equity_vs_vol', 'vix_curve', 'vol_rv']

    models = dict()
    strategies = dict()
    signals = dict()
    outputs = dict()
    perf_summaries = dict()

    def __init__(self):
        pass

    def get_strategy_versions(self):

        # We need something that says here are all the versions of these
        # strategies that we run and track

        strategy_versions = dict()
        strategy_versions['standard'] = dict()
        strategy_versions['portfolio'] = dict()

        # Equity vs volatility
        strategy_versions['standard']['equity_vs_vol'] = dict()
        strategy_versions['standard']['equity_vs_vol']['default'] \
            = self.strategies['equity_vs_vol'].get_default_settings()

        # VIX curve
        strategy_versions['standard']['vix_curve'] = dict()
        strategy_versions['standard']['vix_curve']['default'] \
            = self.strategies['vix_curve'].get_default_settings()

        # Volatility RV
        strategy_versions['portfolio']['vol_rv'] = dict()
        # strategy_versions['portfolio']['vol_rv']['default'] \
        #     = self.strategies['vol_rv'].get_default_settings()
        #
        # # Long-only volatility RV
        # strategy_versions['portfolio']['vol_rv']['long_only'] \
        #     = self.strategies['vol_rv'].get_default_settings()
        # strategy_versions['portfolio']['vol_rv']['long_only']['sell_q'] = None
        #
        # # Short-only volatility RV
        # strategy_versions['portfolio']['vol_rv']['short_only'] \
        #     = self.strategies['vol_rv'].get_default_settings()
        # strategy_versions['portfolio']['vol_rv']['short_only']['buy_q'] = None

        # Volatility RV, wingers
        strategy_versions['portfolio']['vol_rv']['wings'] \
            = self.strategies['vol_rv'].get_default_settings()
        strategy_versions['portfolio']['vol_rv']['long_only']['buy_q'] = 1.0
        strategy_versions['portfolio']['vol_rv']['long_only']['sell_q'] = 0.10

        return strategy_versions

    def initialize_macro_model(self):

        logger.info("initializing macro model...")
        self.models['macro'] = FundamentalMacroModel
        self.outputs['macro'] = self.models['macro'].run()
        logger.info("macro model initialized!")

    def initialize_volatility_factor_model(self, **kwargs):

        logger.info("initializing macro model...")

        # Rolling window for factor model
        minimum_obs = 21
        window_length_days = 512
        update_interval_days = 21
        iv_com = 63
        n_components = 3
        factor_model_start_date = dt.datetime(2010, 1, 1)

        # Universe
        tickers = md.get_etf_vol_universe()

        # Data
        vsm = kwargs.get('vsm', None)
        if vsm is None:
            vsm = VolatilitySurfaceManager()
        vsm.load_data(tickers=tickers, start_date=factor_model_start_date)

        self.models['vol_factor'] = VolatilityFactorModel
        self.models['vol_factor'].initialize_data(vsm=vsm,
                                                  tickers=tickers,
                                                  iv_com=iv_com)

        factor_weights_composite, factor_data_composite, factor_data_oos = \
            self.models['vol_factor'].run(
                minimum_obs=minimum_obs,
                window_length_days=window_length_days,
                update_interval_days=update_interval_days,
                n_components=n_components)

        factor_weights_insample, factor_data_insample = \
            self.models['vol_factor'].run_insample(
                n_components=n_components)

        # Volatility factor model
        self.outputs['vol_factor'] = dict()
        self.outputs['vol_factor']['factor_weights_insample'] = factor_weights_insample
        self.outputs['vol_factor']['factor_data_insample'] = factor_data_insample
        self.outputs['vol_factor']['factor_weights_composite'] = factor_weights_composite
        self.outputs['vol_factor']['factor_data_composite'] = factor_data_composite
        self.outputs['vol_factor']['factor_data_oos'] = factor_data_oos

        logger.info("volatility factor model initialized!")

    def initialize_volatility_rv(self, **kwargs):

        logger.info('initializing volatility RV strategy...')

        vsm = kwargs.get('vsm', None)

        vrv = VolswapRvStrategy()
        vrv.initialize_data(volatility_surface_manager=vsm)
        vrv.process_data()

        rv_iv_signals = vrv.initialize_rv_iv_signals()
        iv_signals = vrv.initialize_iv_signals()
        ts_signals = vrv.initialize_ts_signals()
        rv_signals = vrv.initialize_rv_signals()

        tick_rv_signals = vrv.initialize_tick_rv_signals()

        # macro_signals = vrv.initialize_macro_signals(
        #     factor_weights=self.outputs['vol_factor']['factor_weights_insample'],
        #     macro_factors=self.outputs['macro']['pca_factors_ma'])

        # for signal in macro_signals:
        #     macro_signals[signal] = macro_signals[signal] \
        #         .unstack('ticker') \
        #         .fillna(method='ffill') \
        #         .stack('ticker')

        # add back macro and tick signals
        signal_data = pd.concat([rv_iv_signals,
                                 iv_signals,
                                 ts_signals,
                                 rv_signals], axis=1)
        signal_data_z = vrv.compute_signal_z(signal_data=signal_data,
                                             **kwargs).stack('ticker')
        kwargs['signal_data'] = signal_data

        self.strategies['vol_rv'] = vrv
        self.outputs['vol_rv'] = dict()
        self.outputs['vol_rv']['signal_data'] = signal_data
        self.outputs['vol_rv']['signal_data_z'] = signal_data_z

        logger.info("volatility RV strategy initialized!")

    def initialize_vix_curve(self, **kwargs):

        logger.info('initializing VIX curve strategy...')

        vc = VixCurveStrategy()
        run_sensitivity_analysis = kwargs.get('run_sensitivity_analysis', False)

        # Beginning of semi-clean VIX futures data
        start_date = dt.datetime(2007, 3, 26)
        holding_period_days = 1
        signals_z_cap = 1.0
        vol_target_com = 63
        rolling_beta_com = 63

        # Prep
        vc.initialize_data(vol_futures_series='VX',
                           short_month=1,
                           long_month=5,
                           **kwargs)

        vc.compute_hedge_ratios(rolling_beta_com=rolling_beta_com)

        # Main analysis
        self.outputs['vix_curve'] = vc.compute_master_backtest(
            holding_period_days=holding_period_days,
            signals_z_cap=signals_z_cap,
            vol_target_com=vol_target_com,
            **kwargs)

        # Sensitivity analysis to weights
        if run_sensitivity_analysis:
            num_sims = kwargs.get('num_sims', 1000)
            sigma = kwargs.get('sigma', 2.0)
            sim_perf_percentiles, sim_perf = \
                strat.PortfolioOptimizer.compute_signal_portfolio_sensitivity(
                    strategy=vc,
                    signal_data=self.outputs['vix_curve']['signal_data'],
                    weights=self.outputs['vix_curve']['weights'],
                    num_sims=num_sims,
                    sigma=sigma,
                    signals_z_cap=signals_z_cap,
                    holding_period_days=holding_period_days,
                    vol_target_com=vol_target_com
                )

            self.outputs['vix_curve'].sim_perf_percentiles = sim_perf_percentiles
            self.outputs['vix_curve'].sim_perf = sim_perf

        self.strategies['vix_curve'] = vc
        self.signals['vix_curve'] = self.outputs['vix_curve']['signal_data']

    def initialize_equity_vs_vol(self, **kwargs):

        logger.info('initializing equity versus volatility strategy...')

        vd = VegaVsDeltaStrategy()
        run_sensitivity_analysis = kwargs.get('run_sensitivity_analysis', False)

        # Beginning of semi-clean VIX futures data
        start_date = kwargs.get('start_date', dt.datetime(2007, 3, 26))
        change_window_days_short = 5
        change_window_days_long = 252
        holding_period_days = 1
        signals_z_cap = 1.0
        vol_target_com = 63
        rolling_beta_com = 63
        tc = 0.05

        corr_r_shrinkage = 0.80
        corr_er_shrinkage = 0.60
        er_se_beta_to_er = 0.25
        er_se_beta_to_vol = 1.0

        vd.initialize_data()

        self.outputs['equity_vs_vol'] = vd.compute_master_backtest(
            holding_period_days=holding_period_days,
            vol_target_com=vol_target_com,
            rolling_beta_com=rolling_beta_com,
            signals_z_cap=signals_z_cap,
            transaction_cost_per_unit=tc,
            corr_r_shrinkage=corr_r_shrinkage,
            corr_er_shrinkage=corr_er_shrinkage,
            signal_se_beta_to_er=er_se_beta_to_er,
            signal_se_beta_to_vol=er_se_beta_to_vol)

        if run_sensitivity_analysis:
            num_sims = kwargs.get('num_sims', 1000)
            sigma = kwargs.get('sigma', 2.0)
            self.outputs['equity_vs_vol'].sim_perf_percentiles, \
                self.outputs['equity_vs_vol'].sim_perf = \
                    strat.PortfolioOptimizer.compute_signal_portfolio_sensitivity(
                        strategy=vd,
                        signal_data=self.outputs['vix_curve']['signal_data'],
                        weights=self.outputs['vix_curve']['weights'],
                        num_sims=num_sims,
                        sigma=sigma,
                        signals_z_cap=signals_z_cap,
                        holding_period_days=holding_period_days,
                        vol_target_com=vol_target_com
            )

        self.strategies['equity_vs_vol'] = vd
        self.signals['equity_vs_vol'] = self.outputs['equity_vs_vol']['signal_data']

    def initialize_multistrat_analysis(self, **kwargs):

        weights_version = kwargs.get('weights_version', 'optim_weight')

        multistrat_pnl = pd.DataFrame(
            index=self.outputs['equity_vs_vol']['combined_pnl_net'].index,
            columns=self.strategy_names)

        for strategy in self.strategy_names:
            multistrat_pnl[strategy] = self.outputs[strategy] \
                                           .combined_pnl_net[weights_version]
            multistrat_pnl[strategy] = pd.to_numeric(multistrat_pnl[strategy])

        daily_corr = multistrat_pnl.corr()
        monthly_corr = multistrat_pnl.rolling(
            window=constants.trading_days_per_month).sum().corr()

        multistrat_pnl = multistrat_pnl[np.isfinite(multistrat_pnl).all(axis=1)]

        # Rolling correlation
        weekly_ew_corr = multistrat_pnl.rolling(window=5).sum() \
            .ewm(com=126, min_periods=63).corr()

        self.outputs['multistrat'] = struct
        self.outputs['multistrat'].combined_pnl_net = multistrat_pnl
        self.outputs['multistrat'].daily_corr = daily_corr
        self.outputs['multistrat'].monthly_corr = monthly_corr
        self.outputs['multistrat'].weekly_ew_corr = weekly_ew_corr

    def compute_performance_summaries(self, **kwargs):

        # Compute summary statistics
        self.perf_summaries = dict()

        version = kwargs.get('version', 'optim_weight')
        var_quantile = kwargs.get('var_quantile', 0.005)

        for strategy in ['vix_curve', 'equity_vs_vol']:
            self.perf_summaries[strategy] = dict()

            pnl = self.outputs[strategy]['combined_pnl_net'][version]
            monthly_pnl = pnl.rolling(
                window=int(constants.trading_days_per_month)).sum()

            # Returns and risk
            self.perf_summaries[strategy][
                'monthly_avg_return'] = monthly_pnl.mean()

            # Monthly risk
            self.perf_summaries[strategy]['monthly_avg_vol'] = monthly_pnl.std()
            self.perf_summaries[strategy]['monthly_var_99'] = -monthly_pnl[
                np.isfinite(monthly_pnl)].quantile(var_quantile)

            # Daily risk
            self.perf_summaries[strategy]['daily_avg_vol'] = pnl.std()
            self.perf_summaries[strategy]['daily_var_99'] = -pnl[
                np.isfinite(monthly_pnl)].quantile(var_quantile)

            # Information ratio
            self.perf_summaries[strategy]['info_ratio'] = monthly_pnl.mean() \
                / monthly_pnl.std() * np.sqrt(12.0)

            # Average position size
            self.perf_summaries[strategy]['avg_gross_position'] = \
                self.outputs[strategy]['positions'].abs().mean()

            # Average daily turnover
            self.perf_summaries[strategy]['avg_daily_turnover'] = \
                self.outputs[strategy]['positions'].diff().abs().mean()


    def run_factor_model(self, start_date=None, strategy_names=None, **kwargs):

        vol_target = kwargs.get('vol_target', 0.20)
        vol_handicap = kwargs.get('vol_handicap', 1.5)
        return_window_days = 5

        # Risk factor model

        factor_names = ['MKT', 'MOM', 'SIZE', 'VALUE', 'QUAL', 'CRED', 'DUR',
                        'GOLD', 'OIL']
        long_tickers = ['VTI', 'PDP', 'MGC', 'VTV', 'SPHQ', 'HYG', 'TLT', 'GLD',
                        'USO']
        short_tickers = ['SHV', 'VTI', 'VB', 'VUG', 'SPHB', 'VTI', 'IEI', 'SHV',
                         'SHV']
        return_multipliers = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        hedge_vol_neutral = [0, 1, 1, 1, 1, 1, 0, 0, 0]
        all_tickers = np.unique(short_tickers + long_tickers)

        component_prices = md.get_equity_prices(tickers=all_tickers,
                                                start_date=start_date)[
            'adj_close'] \
            .unstack('ticker')

        factor_returns, factor_daily_returns \
            = stats.ExplicitFactorModel.build_long_short_factors(
                factor_names=factor_names,
                long_tickers=long_tickers,
                short_tickers=short_tickers,
                component_prices=component_prices,
                return_multipliers=return_multipliers,
                hedge_vol_neutral=hedge_vol_neutral)

        strategy_pnl = pd.DataFrame()
        for strategy_name in strategy_names:
            strategy_pnl[strategy_name] = pd.to_numeric(
                self.outputs[strategy_name]['combined_pnl_net']['optim_weight'])

        strategy_pnl_pct = strategy_pnl * vol_target / strategy_pnl.std()\
                           / np.sqrt(252) / vol_handicap
        strategy_pnl.name = strategy_name
        coefs, t_stats, regs = stats.ExplicitFactorModel.compute_exposures(
            target_returns=strategy_pnl_pct.rolling(return_window_days).sum(),
            factor_returns=factor_daily_returns.rolling(return_window_days).sum())
        t_stats = t_stats.abs() / np.sqrt(return_window_days)

        return coefs, t_stats, regs