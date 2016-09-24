import numpy as np
import pandas as pd
import datetime as dt

import struct
from sklearn.decomposition import FactorAnalysis, PCA

import qfl.core.market_data as md
import qfl.core.calcs as calcs
from qfl.core.market_data import VolatilitySurfaceManager, RealizedVolatilityManager
from qfl.utilities.statistics import RollingFactorModel


class Model(object):

    settings = None
    raw_data = None
    data = None
    calc = None

    @classmethod
    def initialize_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def run(cls, **kwargs):
        raise NotImplementedError


class VolatilityFactorModel(Model):

    @classmethod
    def initialize_data(cls, **kwargs):

        cls.settings = struct
        cls.settings.default_start_date = dt.datetime(2010, 1, 1)

        cls.settings.start_date = kwargs.get('start_date',
                                             cls.settings.default_start_date)

        iv_com = kwargs.get('iv_com', 21)
        tickers = kwargs.get('tickers', None)
        ivol_field = kwargs.get('ivol_field', 'iv_3m')
        if tickers is None:
            raise ValueError("Must include 'tickers' argument!")
        vsm = kwargs.get("volatility_surface_manager", None)
        if vsm is None:
            vsm = VolatilitySurfaceManager()
            vsm.load_data(tickers, cls.settings.start_date)

        iv = vsm.get_data(tickers=tickers,
                          fields=[ivol_field],
                          start_date=cls.settings.start_date)
        iv = iv[ivol_field].unstack('ticker')

        sp = md.get_equity_prices(tickers=tickers,
                                  start_date=cls.settings.start_date)
        sp = sp['adj_close'].unstack('ticker')

        # iv_c, normal_tests = calcs.clean_implied_vol_data(
        #     ivol=iv,
        #     tickers=tickers,
        #     ref_ivol_ticker='SPY',
        #     stock_prices=sp)\
        #     .fillna(method='ffill')

        iv_c = iv

        # PCA on rolling changes
        iv_chg = iv_c.diff(1).ewm(com=iv_com).mean()
        iv_chg_z = (iv_chg - iv_chg.mean()) / iv_chg.std()

        cls.raw_data = iv
        cls.data = struct
        cls.data.iv = iv_c
        cls.data.sp = sp
        cls.data.vsm = vsm
        cls.data.iv_chg = iv_chg
        cls.data.iv_chg_z = iv_chg_z

    @classmethod
    def run_insample(cls, **kwargs):

        iv_chg_z = cls.data.iv_chg_z.copy(deep=True)
        n_components = kwargs.get('n_components', 3)

        iv_chg_z_ = iv_chg_z[np.isfinite(iv_chg_z).all(axis=1)]
        fa = FactorAnalysis(n_components=n_components).fit(iv_chg_z_)
        factor_data_insample = pd.DataFrame(index=iv_chg_z_.index,
                                            data=fa.transform(iv_chg_z_))
        factor_weights_insample = pd.DataFrame(index=iv_chg_z_.columns,
                                               data=fa.components_.transpose())
        if factor_weights_insample.loc['SPY', 0] < 0:
            factor_weights_insample[0] *= -1.0
            factor_data_insample[0] *= -1.0

        return factor_weights_insample, factor_data_insample

    @classmethod
    def run(cls, **kwargs):

        minimum_obs = kwargs.get('minimum_obs', 21)
        window_length_days = kwargs.get('window_length_days', 512)
        update_interval_days = kwargs.get('update_interval_days', 21)
        n_components = kwargs.get('n_components', 3)

        factor_weights, factor_data, factor_data_oos = \
            RollingFactorModel.run(data=cls.data.iv_chg_z,
                                   minimum_obs=minimum_obs,
                                   window_length_days=window_length_days,
                                   update_interval_days=update_interval_days,
                                   n_components=n_components)

        return factor_weights, factor_data, factor_data_oos


