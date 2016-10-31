import pandas as pd
import numpy as np
import qfl.utilities.basic_utilities as utils

import qfl.core.calcs as calcs
import qfl.core.market_data as md


class SecurityMaster(object):

    # Data stores the
    _data = None
    _json_data = None
    _date_cols = list()

    def __init__(self, data=None, attribute_columns=None):

        # super(SecurityMaster, self).__init__(data=data)

        if attribute_columns is None:
            attribute_columns = data.columns
        self._json_data = utils.json_column_from_columns(
            df=data,
            columns=attribute_columns,
            new_col_name='attributes')

        self._data = data

        for col in data:
            if utils.is_date_type(data[col].values[0]):
                self._date_cols.append(col)

    def expand_attributes(self):
        df = utils.df_columns_from_json_column(df=self._json_data,
                                               json_col_name='attributes')
        for col in self._date_cols:
            df[col] = pd.to_datetime(df[col])
        return df

    def compress_attributes(self, attribute_columns):
        if attribute_columns is None:
            attribute_columns = self._data.columns
        return utils.json_column_from_columns(df=self._data,
                                              columns=attribute_columns,
                                              new_col_name='attributes')


class MarkingEngine(object):

    @staticmethod
    def get_equity_option_implied_volatility_marks(security_master=None,
                                                   start_date=None,
                                                   end_date=None,
                                                   vsm=None):

        df = security_master.expand_attributes()

        unique_und = np.unique(df['underlying'])
        df_dict = dict()
        for und in unique_und:
            print(und)

            df_und = df[df['underlying'] == und]
            mats = pd.to_datetime(
                df_und['maturity_date'].values.tolist()).tolist()
            strikes = df_und['strike'].values.tolist()

            df_dict[und] = vsm.get_fixed_maturity_date_vol_by_strike(
                ticker=und,
                strikes=strikes,
                maturity_dates=mats,
                start_date=start_date,
                end_date=end_date
            ).stack(level=['strike', 'maturity_date'])

        ivol = pd.concat(df_dict).reset_index(level=0, drop=True)

        return ivol

    @staticmethod
    def compute_instrument_outputs(instrument_type=None,
                                   security_master=None,
                                   start_date=None,
                                   end_date=None,
                                   vsm=None,
                                   inputs=None):

        if inputs is None:
            inputs = MarkingEngine.get_instrument_inputs(
                instrument_type=instrument_type,
                security_master=security_master,
                start_date=start_date,
                end_date=end_date,
                vsm=vsm
            )

        outputs = pd.DataFrame()
        if instrument_type == 'equity_option':
            outputs = MarkingEngine.mark_equity_options(
                inputs=inputs,
                start_date=start_date,
                end_date=end_date,
                security_master=security_master,
                vsm=vsm
            )

        return outputs

    @staticmethod
    def get_instrument_inputs(instrument_type=None,
                              security_master=None,
                              start_date=None,
                              end_date=None,
                              vsm=None):

        # Security_master should be a DataFrame with colummns including
        # instrument, instrument_type and attributes (JSON)

        df = security_master.expand_attributes()
        df = df[df['instrument_type'] == instrument_type]

        inputs = pd.DataFrame()

        if instrument_type == 'equity_option':

            inputs['iv'] = MarkingEngine\
                .get_equity_option_implied_volatility_marks(
                security_master=security_master,
                start_date=start_date,
                end_date=end_date,
                vsm=vsm)

            unique_und = [str(u) for u in np.unique(df['underlying'])]
            stock_prices = md.get_equity_prices(tickers=unique_und,
                                                start_date=start_date,
                                                end_date=end_date)['adj_close']

            cols = ['ticker', 'date', 'strike', 'maturity_date']
            inputs = pd.merge(left=inputs.reset_index(),
                              right=stock_prices.reset_index(),
                              on=['ticker', 'date']).set_index(cols)

            # inputs['strike'] = inputs.index.get_level_values('strike')
            inputs = inputs.rename(columns={'adj_close': 'spot'})

            cols = ['instrument',
                    'option_type',
                    'strike',
                    'ticker',
                    'maturity_date']

            inputs = pd.merge(
                left=inputs.reset_index(),
                right=df.reset_index()\
                    .rename(columns={'underlying': 'ticker'})[cols],
                on=['strike', 'maturity_date', 'ticker']
                ).set_index(['date', 'instrument'], drop=True)

            # TODO: fix
            inputs['risk_free'] = 0.0
            inputs['dividend_yield'] = 0.0

            # TODO: support multiple calendars
            dates = pd.Series(inputs.index.get_level_values('date'))
            inputs['tenor_in_days'] = utils.networkdays(
                start_date=dates,
                end_date=inputs['maturity_date']
            )

            # Remove after maturity
            inputs = inputs[inputs['tenor_in_days'] >= 0]

        return inputs

    @staticmethod
    def mark_volswaps(positions=None, inputs=None, dates=None, fields=None):

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
        positions['quantity_gross'] = positions['quantity'].abs()

        return positions

    @staticmethod
    def mark_equity_options(inputs=None,
                            fields=None,
                            start_date=None,
                            end_date=None,
                            security_master=None,
                            vsm=None):

        if fields is None:
            fields = ['price', 'delta_shares', 'delta_dollar',
                      'vega', 'gamma_shares', 'gamma_pct_dollar', 'theta']

        if inputs is None:
            inputs = MarkingEngine.get_instrument_inputs(
                instrument_type='equity_option',
                start_date=start_date,
                end_date=end_date,
                security_master=security_master,
                vsm=vsm
            )

        outputs = pd.DataFrame(index=inputs.index, columns=fields)

        if 'price' in fields:
            outputs['price'] = calcs.black_scholes_price(
                spot=inputs['spot'],
                strike=inputs['strike'],
                tenor_in_days=inputs['tenor_in_days'],
                ivol=inputs['iv'] / 100.0,
                div=inputs['dividend_yield'],
                div_type='yield',
                risk_free=inputs['risk_free'],
                option_type=inputs['option_type'])

        if 'delta_shares' in fields or 'delta_dollar' in fields:
            outputs['delta_shares'] = calcs.black_scholes_delta(
                spot=inputs['spot'],
                strike=inputs['strike'],
                tenor_in_days=inputs['tenor_in_days'],
                ivol=inputs['iv'] / 100.0,
                div=inputs['dividend_yield'],
                div_type='yield',
                risk_free=inputs['risk_free'],
                option_type=inputs['option_type'])

        if 'delta_dollar' in fields:
            outputs['delta_dollar'] = outputs['delta_shares'] * inputs['spot']

        if 'vega' in fields:
            outputs['vega'] = calcs.black_scholes_vega(
                spot=inputs['spot'],
                strike=inputs['strike'],
                tenor_in_days=inputs['tenor_in_days'],
                ivol=inputs['iv'] / 100.0,
                div=inputs['dividend_yield'],
                div_type='yield',
                risk_free=inputs['risk_free'],
                option_type=inputs['option_type'])

        return outputs

