import pandas as pd
import numpy as np
import datetime as dt

import qfl.utilities.basic_utilities as utils


def backtest_equity_portfolio(portfolio_target_weights=None,
                              stock_prices=None,
                              start_date=None,
                              end_date=dt.datetime.today(),
                              rebalance_frequency_days=63):

    """
    :param portfolio_target_weights:
    :param stock_prices: DataFrame indexed on "date" and "ticker"
    :param start_date:
    :param end_date:
    :param rebalance_frequency_days:
    :return:
    """

    daily_returns = stock_prices / stock_prices.shift(1) - 1
    dates = daily_returns.index
    daily_returns = daily_returns.fillna(value=0)

    # Continuously rebalanced portfolio
    portfolio_daily_returns = pd.DataFrame(index=daily_returns.index,
                                           columns=["daily_return"])
    portfolio_daily_returns['daily_return'] = 0
    for ticker in portfolio_target_weights.keys():
        portfolio_daily_returns['daily_return'] += daily_returns[ticker] \
                                                   * portfolio_target_weights[
                                                       ticker]

    cumulative_portfolio_returns = pd.DataFrame(index=dates)
    cumulative_portfolio_returns['continuous'] = \
        (1 + portfolio_daily_returns).cumprod() - 1

    # Properly rebalanced portfolio
    portfolio_weights = pd.DataFrame(index=dates,
                                     columns=portfolio_target_weights.keys())
    days_since_rebalance = pd.Series(index=dates)
    portfolio_weights.iloc[0] = portfolio_target_weights
    days_since_rebalance.iloc[0] = 0
    for i in range(1, len(dates)):
        date = dates[i]
        days_since_rebalance.iloc[i] = days_since_rebalance.iloc[i - 1] + 1
        if days_since_rebalance.loc[date] >= rebalance_frequency_days:
            days_since_rebalance.loc[date] = 0
            portfolio_weights.loc[date] = portfolio_target_weights
        portfolio_weights.iloc[i] = portfolio_weights.iloc[i - 1] \
                                    * (1 + daily_returns.loc[dates[i]])
        portfolio_weights.iloc[i] /= portfolio_weights.iloc[i].sum()

    portfolio_weights = portfolio_weights.copy(deep=True)
    portfolio_returns = (portfolio_weights * daily_returns).sum(axis=1)
    portfolio_returns = portfolio_returns.fillna(value=0)
    cumulative_portfolio_returns['rebalance'] = \
        (1 + portfolio_returns).cumprod() - 1

    return cumulative_portfolio_returns, \
           portfolio_returns, \
           portfolio_weights


def build_positions_from_transactions(end_date=None,
                                      base_currency='USD',
                                      calendar_name=None,
                                      cash_flows=None,
                                      transactions=None,
                                      security_income=None,
                                      other_adjustments=None,
                                      interest_rates=None,
                                      asset_prices=None,
                                      price_field='last_price'):

    # Asset id for base currency cash
    base_asset = 'cash_' + base_currency

    # Default start and end dates
    start_date = cash_flows['date'].min()

    if end_date is None:
        end_date = transactions['date'].max()

    # Columns
    cols = ['date', 'quantity', 'market_value']

    # Dates
    dates = utils.DateUtils.get_business_date_range(start_date,
                                                    end_date,
                                                    calendar_name)
    start_date = dates[0]

    # Initial positions
    initial_positions = pd.DataFrame(index=[pd.Series(base_asset)],
                                     columns=cols)
    initial_positions.index.names = ['asset_id']
    initial_positions.loc[base_asset, ['quantity', 'market_value']] \
        = cash_flows[cash_flows['date'] == start_date]['market_value'].values[0]
    initial_positions.loc[base_asset, 'date'] = start_date

    # Iterate over dates
    positions_dict = dict()
    positions_dict[0] = initial_positions
    for t in range(1, len(dates)):

        # Starting point: carry over positions
        positions_dict[t] = positions_dict[t-1].copy(deep=True)
        positions_dict[t]['date'] = dates[t]

        # Cash balances accrue interest
        elapsed_days = (dates[t] - dates[t-1]).days
        daily_rate = interest_rates[(interest_rates.index >= dates[t - 1])
                                  & (interest_rates.index < dates[t])].mean()

        # Carry over prior value
        if np.isnan(daily_rate):
            daily_rate = prev_daily_rate
        prev_daily_rate = daily_rate

        positions_dict[t].loc[base_asset, ['quantity', 'market_value']] \
            *= (1 + daily_rate) ** (elapsed_days / 365.0)

        # New cash flows
        # TODO: what if the cash flow is in a different currency
        cf = cash_flows[(cash_flows['date'] <= dates[t])
                      & (cash_flows['date'] > dates[t - 1])]
        positions_dict[t].loc[base_asset, ['quantity', 'market_value']]\
            += cf['market_value'].sum()

        # Security income
        # TODO: what if the cash flow is in a different currency
        si = security_income[(security_income['date'] <= dates[t])
                           & (security_income['date'] > dates[t - 1])]
        positions_dict[t].loc[base_asset, ['quantity', 'market_value']]\
            += si['market_value'].sum()

        # Adjustments
        # TODO: what if the cash flow is in a different currency
        oa = other_adjustments[(other_adjustments['date'] <= dates[t])
                             & (other_adjustments['date'] > dates[t - 1])]
        positions_dict[t].loc[base_asset, ['quantity', 'market_value']]\
            += oa['market_value'].sum()

        # New transactions
        tt = transactions[(transactions['date'] <= dates[t])
                        & (transactions['date'] > dates[t - 1])]
        if len(tt) > 0:

            # Debit cash account for transactions
            positions_dict[t].loc[base_asset, ['quantity', 'market_value']]\
                -= tt['market_value'].sum()

            # Changes in existing positions
            ind = tt.index[tt['asset_id'].isin(positions_dict[t].index)]
            if len(ind) > 0:
                asset_codes = tt.loc[ind, 'asset_id']
                positions_dict[t].loc[asset_codes, ['quantity', 'market_value']] \
                    += tt.loc[ind, ['quantity', 'market_value']].values
            # New positions
            new_ind = tt.index[~tt['asset_id'].isin(positions_dict[t].index)]
            if len(new_ind) > 0:
                asset_codes = tt.loc[new_ind, 'asset_id']
                new_df = pd.DataFrame(index=asset_codes,
                                      columns=cols,
                                      data=tt.loc[new_ind, cols]
                                      .values)
                positions_dict[t] = positions_dict[t].append(new_df)

        # Mark the book
        price_data_date = asset_prices[
            asset_prices.index.get_level_values('date') == dates[t]]\
            .reset_index()
        price_data_date.index = price_data_date['asset_id']
        ind = positions_dict[t].index[
            positions_dict[t].index.isin(price_data_date.index)]
        positions_dict[t].loc[ind, 'market_value'] = \
            positions_dict[t].loc[ind, 'quantity'] \
            * price_data_date.loc[ind, price_field]

    positions = pd.concat(positions_dict).reset_index()
    positions.index = [positions['date'], positions['asset_id']]
    positions = positions.rename(columns={'level_0': 'date_index'})

    # Join positions data to price for visual inspection
    positions = pd.merge(
        left=positions,
        right=pd.DataFrame(asset_prices['Price']),
        left_index=True,
        right_index=True,
        how='left')

    # Drop stuff we have none of
    drop_ind = positions.index[
        (positions.index.get_level_values('asset_id') != base_asset)
        & (positions['quantity'] == 0.0)]
    positions = positions.drop(drop_ind)

    # Set price for cash equal to 1
    cash_ind = positions.index[positions.index.get_level_values(
        'asset_id') == base_asset]
    positions.loc[cash_ind, 'Price'] = 1.0
    del positions['asset_id']
    del positions['date']

    account_performance = compute_asset_performance_summary(positions,
                                                            cash_flows)
    return positions, account_performance


def compute_asset_performance_summary(positions=None,
                                      cash_flows=None):

    account_performance = pd.DataFrame(positions['market_value']
                                       .groupby(level='date').sum())

    account_performance['daily_flows'] = cash_flows.groupby('date') \
        ['market_value'].sum()
    account_performance['daily_flows'] = account_performance['daily_flows'] \
        .fillna(value=0)
    account_performance['pnl'] = account_performance['market_value'].diff(1) \
                                 - account_performance['daily_flows']
    account_performance['pnl_pct'] = account_performance['pnl'] \
                                     / account_performance[
                                         'market_value'].shift(1)
    account_performance['cum_pnl_pct'] = (1 + account_performance['pnl_pct']) \
                                             .cumprod() - 1
    account_performance['cum_pnl_gbp'] = account_performance['pnl'].cumsum()

    return account_performance


