""" HW4 : Market simulator."""
import pandas
import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def compute_portvals(orders_file="./orders/orders.csv", start_val=10000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'],
                         usecols=['Date', 'Symbol', 'Order', 'Shares'])
    orders.sort_index(inplace=True)
    symbols = np.array(orders.Symbol.unique()).tolist()
    start_date = pandas.to_datetime(orders.index[0])
    end_date = pandas.to_datetime(orders.index[-1])
    # start_date = dt.datetime(2011, 1, 10)
    # end_date = dt.datetime(2011, 12, 20)
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    pr = prices.index
    rows = orders.iterrows()
    series = pd.Series(start_val, pr)
    prices['Portfolio'], prices['Cash'] = series, series
    for symbol in symbols:
        prices[symbol + ' Shares'] = pd.Series(0, pr)
    for left, row in rows:
        left = left.date()
        left = left.strftime("%Y-%m-%d")
        order_var = row['Order']
        symbol = row['Symbol']
        share = row['Shares']
        symShare = symbol + ' Shares'
        shares = prices.loc[left, symShare]
        cash = prices.loc[left, symbol] * share
        if order_var == 'SELL':
            shares -= share
            cash *= (1 - impact)
            cash -= commission
            prices.loc[left:, 'Cash'] += cash
        else:
            shares += share
            cash *= (1 + impact)
            cash += commission
            prices.loc[left:, 'Cash'] -= cash
        prices.loc[left:, symShare] = shares
    rows = prices.iterrows()
    for left, row in rows:
        i = 0
        for symbol in symbols:
            symShare, sym = symbol + ' Shares', row[symbol]
            i += prices.loc[left, symShare] * sym
            prices.loc[left, 'Portfolio'] = prices.loc[left, 'Cash'] + i
    return prices.loc[:, 'Portfolio'], start_date, end_date


def GetSpyDf(start, end):
    start_date = start
    end_date = end
    portvals = get_data(['SPY'], pd.date_range(start_date, end_date))
    portvals = portvals[['SPY']]  # remove SPY
    portvals.loc[:, 'SPY'] *= 773.251677956
    return portvals


def RunCode():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    # start_date = start
    # end_date = end
    # portvals_spy = GetSpyDf(start, end)
    # portvals_spy = portvals_spy.loc[:, 'SPY']
    # daily_returns_spy = portvals_spy.pct_change(1)
    # std_daily_spy = daily_returns_spy.std()
    # sr_df_spy = daily_returns_spy - 0.0
    # sr_spy = np.sqrt(252) * (sr_df_spy.mean() / std_daily_spy)
    # daily_returns_spy = daily_returns_spy.mean()
    #
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 10000) / 10000, daily_returns,
    #                                                        std_daily, sr]
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [(portvals_spy.iloc[-1] - 10000) / 10000,
    #                                                                        daily_returns_spy,
    #                                                                        std_daily_spy, sr_spy]



    # Create Portfolio Comprised of SPY ------------------------------------------------------------------
    of = "./orders/FINAL_PROJECT_ORDERS_SPY.csv"
    sv = 10000

    # Process orders
    portvals, start, end = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    start_date = start
    end_date = end
    daily_returns = portvals.pct_change(1)
    std_daily = daily_returns.std()
    sr_df = daily_returns - 0.0
    sr = np.sqrt(252) * (sr_df.mean() / std_daily)
    daily_returns = daily_returns.mean()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 1000000) / 1000000, daily_returns,
                                                           std_daily, sr]

    print(f"Start Date:  \t\t\t{start_date}")
    print(f"End Date:    \t\t\t{end_date}")

    print("")
    print(f"Sharpe Ratio of Portfolio comprised of $SPY: \t\t{sharpe_ratio}")
    print(f"Cumulative Return of Portfolio comprised of $SPY: \t{cum_ret}")
    print(f"Standard Deviation of Portfolio comprised of $SPY: \t{std_daily_ret}")
    print(f"Average Daily Return of Portfolio comprised of $SPY: \t{avg_daily_ret}")
    print(f"Final Portfolio Value: \t\t{portvals[-1]}")

    plot_data(portvals, "SPY")

    # Create Portfolio Comprised of SPXL ------------------------------------------------------------------
    of = "./orders/FINAL_PROJECT_ORDERS_SPXL.csv"
    sv = 10000

    # Process orders
    portvals, start, end = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    daily_returns = portvals.pct_change(1)
    std_daily = daily_returns.std()
    sr_df = daily_returns - 0.0
    sr = np.sqrt(252) * (sr_df.mean() / std_daily)
    daily_returns = daily_returns.mean()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 1000000) / 1000000, daily_returns,
                                                           std_daily, sr]

    print("")
    print(f"Sharpe Ratio of Portfolio comprised of $SPXL: \t\t{sharpe_ratio}")
    print(f"Cumulative Return of Portfolio comprised of $SPXL: \t{cum_ret}")
    print(f"Standard Deviation of Portfolio comprised of $SPXL: \t{std_daily_ret}")
    print(f"Average Daily Return of Portfolio comprised of $SPXL: \t{avg_daily_ret}")
    print(f"Final Portfolio Value: \t\t{portvals[-1]}")

    plot_data(portvals, "SPXL")

    # Create Portfolio Comprised of QQQ ------------------------------------------------------------------
    of = "./orders/FINAL_PROJECT_ORDERS_QQQ.csv"
    sv = 10000

    # Process orders
    portvals, start, end = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    daily_returns = portvals.pct_change(1)
    std_daily = daily_returns.std()
    sr_df = daily_returns - 0.0
    sr = np.sqrt(252) * (sr_df.mean() / std_daily)
    daily_returns = daily_returns.mean()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 1000000) / 1000000, daily_returns,
                                                           std_daily, sr]

    print("")
    print(f"Sharpe Ratio of Portfolio comprised of $QQQ: \t\t{sharpe_ratio}")
    print(f"Cumulative Return of Portfolio comprised of $QQQ: \t{cum_ret}")
    print(f"Standard Deviation of Portfolio comprised of $QQQ: \t{std_daily_ret}")
    print(f"Average Daily Return of Portfolio comprised of $QQQ: \t{avg_daily_ret}")
    print(f"Final Portfolio Value: \t\t{portvals[-1]}")

    plot_data(portvals, "QQQ")

    # Create Portfolio Comprised of TQQQ ------------------------------------------------------------------
    of = "./orders/FINAL_PROJECT_ORDERS_TQQQ.csv"
    sv = 10000

    # Process orders
    portvals, start, end = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    daily_returns = portvals.pct_change(1)
    std_daily = daily_returns.std()
    sr_df = daily_returns - 0.0
    sr = np.sqrt(252) * (sr_df.mean() / std_daily)
    daily_returns = daily_returns.mean()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 1000000) / 1000000, daily_returns,
                                                           std_daily, sr]
    print("")
    print(f"Sharpe Ratio of Portfolio comprised of $TQQQ: \t\t{sharpe_ratio}")
    print(f"Cumulative Return of Portfolio comprised of $TQQQ: \t{cum_ret}")
    print(f"Standard Deviation of Portfolio comprised of $TQQQ: \t{std_daily_ret}")
    print(f"Average Daily Return of Portfolio comprised of $TQQQ: \t{avg_daily_ret}")
    print(f"Final Portfolio Value: \t\t{portvals[-1]}")

    plot_data(portvals, "TQQQ")

    # Create Portfolio Comprised of SQQQ ------------------------------------------------------------------
    of = "./orders/FINAL_PROJECT_ORDERS_SQQQ.csv"
    sv = 10000

    # Process orders
    portvals, start, end = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    daily_returns = portvals.pct_change(1)
    std_daily = daily_returns.std()
    sr_df = daily_returns - 0.0
    sr = np.sqrt(252) * (sr_df.mean() / std_daily)
    daily_returns = daily_returns.mean()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 1000000) / 1000000, daily_returns,
                                                           std_daily, sr]
    print("")
    print(f"Sharpe Ratio of Portfolio comprised of $SQQQ: \t\t{sharpe_ratio}")
    print(f"Cumulative Return of Portfolio comprised of $SQQQ: \t{cum_ret}")
    print(f"Standard Deviation of Portfolio comprised of $SQQQ: \t{std_daily_ret}")
    print(f"Average Daily Return of Portfolio comprised of $SQQQ: \t{avg_daily_ret}")
    print(f"Final Portfolio Value: \t\t{portvals[-1]}")

    plot_data(portvals, "SQQQ")

    # Create Portfolio Comprised of SQQQ Hedging with SPXL and TQQQ ---------------------------------------------------
    of = "./orders/FINAL_PROJECT_ORDERS_SQQQHEDGETQQQSPXL.csv"
    sv = 10000

    # Process orders
    portvals, start, end = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    daily_returns = portvals.pct_change(1)
    std_daily = daily_returns.std()
    sr_df = daily_returns - 0.0
    sr = np.sqrt(252) * (sr_df.mean() / std_daily)
    daily_returns = daily_returns.mean()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 1000000) / 1000000, daily_returns,
                                                           std_daily, sr]
    print("")
    print(f"Sharpe Ratio of Portfolio comprised of $SQQQ, $TQQQ and $SPXL: \t\t{sharpe_ratio}")
    print(f"Cumulative Return of Portfolio comprised of $SQQQ, $TQQQ and $SPXL: \t{cum_ret}")
    print(f"Standard Deviation of Portfolio comprised of $SQQQ, $TQQQ and $SPXL: \t{std_daily_ret}")
    print(f"Average Daily Return of Portfolio comprised of $SQQQ, $TQQQ and $SPXL: \t{avg_daily_ret}")
    print(f"Final Portfolio Value: \t\t{portvals[-1]}")

    plot_data(portvals, "SQQQ Hedging w/ TQQQ and SPXL")

    # Create Portfolio Comprised of SQQQ for just 2022 ---------------------------------------------------
    of = "./orders/FINAL_PROJECT_ORDERS_SQQQ2022.csv"
    sv = 10000

    # Process orders
    portvals, start, end = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    daily_returns = portvals.pct_change(1)
    std_daily = daily_returns.std()
    sr_df = daily_returns - 0.0
    sr = np.sqrt(252) * (sr_df.mean() / std_daily)
    daily_returns = daily_returns.mean()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 1000000) / 1000000, daily_returns,
                                                           std_daily, sr]
    print("")
    print(f"Sharpe Ratio of Portfolio comprised of $SQQQ in 2022 ONLY: \t\t{sharpe_ratio}")
    print(f"Cumulative Return of Portfolio comprised of $SQQQ in 2022 ONLY: \t{cum_ret}")
    print(f"Standard Deviation of Portfolio comprised of $SQQQ in 2022 ONLY: \t{std_daily_ret}")
    print(f"Average Daily Return of Portfolio comprised of $SQQQ in 2022 ONLY: \t{avg_daily_ret}")
    print(f"Final Portfolio Value: \t\t{portvals[-1]}")

    plot_data(portvals, "SQQQ in 2022")

    # Start making the I-bond combination portfolios here ---------------------------------------------------

    # 80% $SPY and 20% I-Bonds -------------------------------------------------------------------
    of = "./orders/FINAL_PROJECT_ORDERS_80SPY20IBONDS.csv"
    sv = 10000

    # Process orders
    portvals, start, end = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    daily_returns = portvals.pct_change(1)
    std_daily = daily_returns.std()
    sr_df = daily_returns - 0.0
    sr = np.sqrt(252) * (sr_df.mean() / std_daily)
    daily_returns = daily_returns.mean()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 1000000) / 1000000, daily_returns,
                                                           std_daily, sr]
    print("")
    print(f"Sharpe Ratio of Portfolio comprised of $SPY and IBONDS: \t\t{sharpe_ratio}")
    print(f"Cumulative Return of Portfolio comprised of $SPY and IBONDS: \t{cum_ret}")
    print(f"Standard Deviation of Portfolio comprised of $SPY and IBONDS: \t{std_daily_ret}")
    print(f"Average Daily Return of Portfolio comprised of $SPY and IBONDS: \t{avg_daily_ret}")
    print(f"Final Portfolio Value: \t\t{portvals[-1]}")

    plot_data(portvals, "SPY + I-Bonds")

    # 50% $SPXL and 50% I-Bonds -------------------------------------------------------------------
    of = "./orders/FINAL_PROJECT_ORDERS_50SPXL50IBONDS.csv"
    sv = 10000

    # Process orders
    portvals, start, end = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    daily_returns = portvals.pct_change(1)
    std_daily = daily_returns.std()
    sr_df = daily_returns - 0.0
    sr = np.sqrt(252) * (sr_df.mean() / std_daily)
    daily_returns = daily_returns.mean()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 1000000) / 1000000, daily_returns,
                                                           std_daily, sr]
    print("")
    print(f"Sharpe Ratio of Portfolio comprised of $SPXL and IBONDS: \t\t{sharpe_ratio}")
    print(f"Cumulative Return of Portfolio comprised of $SPXL and IBONDS: \t{cum_ret}")
    print(f"Standard Deviation of Portfolio comprised of $SPXL and IBONDS: \t{std_daily_ret}")
    print(f"Average Daily Return of Portfolio comprised of $SPXL and IBONDS: \t{avg_daily_ret}")
    print(f"Final Portfolio Value: \t\t{portvals[-1]}")

    plot_data(portvals, "SPXL + I-Bonds")

    # 20% $SQQQ and 80% I-Bonds -------------------------------------------------------------------
    of = "./orders/FINAL_PROJECT_ORDERS_20SQQQ80IBONDS.csv"
    sv = 10000

    # Process orders
    portvals, start, end = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    daily_returns = portvals.pct_change(1)
    std_daily = daily_returns.std()
    sr_df = daily_returns - 0.0
    sr = np.sqrt(252) * (sr_df.mean() / std_daily)
    daily_returns = daily_returns.mean()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 1000000) / 1000000, daily_returns,
                                                           std_daily, sr]
    print("")
    print(f"Sharpe Ratio of Portfolio comprised of $SPXL and IBONDS: \t\t{sharpe_ratio}")
    print(f"Cumulative Return of Portfolio comprised of $SPXL and IBONDS: \t{cum_ret}")
    print(f"Standard Deviation of Portfolio comprised of $SPXL and IBONDS: \t{std_daily_ret}")
    print(f"Average Daily Return of Portfolio comprised of $SPXL and IBONDS: \t{avg_daily_ret}")
    print(f"Final Portfolio Value: \t\t{portvals[-1]}")

    plot_data(portvals, "SQQQ + I-Bonds")

    # 100% I-Bonds -------------------------------------------------------------------
    of = "./orders/FINAL_PROJECT_ORDERS_100IBONDS.csv"
    sv = 10000

    # Process orders
    portvals, start, end = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    daily_returns = portvals.pct_change(1)
    std_daily = daily_returns.std()
    sr_df = daily_returns - 0.0
    sr = np.sqrt(252) * (sr_df.mean() / std_daily)
    daily_returns = daily_returns.mean()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [(portvals.iloc[-1] - 1000000) / 1000000, daily_returns,
                                                           std_daily, sr]
    print("")
    print(f"Sharpe Ratio of Portfolio comprised of IBONDS: \t\t{sharpe_ratio}")
    print(f"Cumulative Return of Portfolio comprised of IBONDS: \t{cum_ret}")
    print(f"Standard Deviation of Portfolio comprised of IBONDS: \t{std_daily_ret}")
    print(f"Average Daily Return of Portfolio comprised of IBONDS: \t{avg_daily_ret}")
    print(f"Final Portfolio Value: \t\t{portvals[-1]}")

    plot_data(portvals, "I-Bonds")


if __name__ == "__main__":
    RunCode()
