import pandas as pd
import numpy as np

import qfl.core.calcs as calcs


class MarkingEngine(object):

    @staticmethod
    def mark_volswaps(positions=None, dates=None):

        # Positions is a dataframe with columns ['iv', 'rv', 'strike',
        # 'maturity_date', 'start_date', 'date'

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

        return positions

    @staticmethod
    def mark_options():

        pass

