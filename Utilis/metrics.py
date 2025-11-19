import numpy as np
import pandas as pd


def calculate_downside_deviation(returns, target_return=0, annualization_factor=252):
    downside_diff = returns - target_return
    downside_diff[downside_diff > 0] = 0
    squared_diffs = downside_diff ** 2
    downside_variance = np.mean(squared_diffs)
    annualized_downside_deviation = np.sqrt(downside_variance) * np.sqrt(annualization_factor)
    return annualized_downside_deviation


def calculate_sortino_ratio(returns, risk_free_rate=0, target_return=0, annualization_factor=252):
    mean_return = returns.mean() * annualization_factor
    downside_dev = calculate_downside_deviation(returns, target_return, annualization_factor)
    if downside_dev == 0:
        return np.nan

    sortino = (mean_return - risk_free_rate) / downside_dev
    return sortino


def calculate_mdd(returns):
    returns_array = np.asarray(returns)
    cumulative_returns = np.cumprod(1 + returns_array)
    high_water_mark = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - high_water_mark) / high_water_mark
    mdd = np.min(drawdown)
    return mdd

def calculate_percent_positive_returns(returns):
    positive_periods = (returns > 0).sum()
    total_periods = len(returns)
    return positive_periods / total_periods

def report_metrics(ret):
    res = {}
    res['annual_ret'] = np.mean(ret) * 252
    res['annual_std'] = np.std(ret) * np.sqrt(252)
    res['annual_sharpe'] = (np.mean(ret) / np.std(ret)) * np.sqrt(252)
    res['annual_dd']=calculate_downside_deviation(ret)
    res['annual_sortino'] = calculate_sortino_ratio(ret)
    res['annual_mdd'] = calculate_mdd(ret)
    res['annual_%_of_+_RET'] = calculate_percent_positive_returns(ret)


    return res