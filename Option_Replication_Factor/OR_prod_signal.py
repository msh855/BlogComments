import pandas as pd
import numpy as np
from scipy.stats import norm

from myScripts.download_and_prepare_data import get_ccy10y_spreads, get_ccy2y_spreads, get_commodity_prices, \
    relative_spread, get_data


# Options replication factor
def realised_vol(spot_return, decay):
    spot_momentum = spot_return.ewm(com=decay / (1.0 - decay), adjust=False).mean()
    adj_return = spot_return - spot_momentum
    adj_returnsqr = 252 * adj_return * adj_return
    variance_ewma = adj_returnsqr.ewm(com=decay / (1.0 - decay), adjust=False).mean()

    return np.sqrt(variance_ewma)


def model_vol(real_vol, imp_vol):
    '''
    Builds the model volatility, which is the pointwise maximum of the realized volatility and the implied volatility.
    '''
    df = pd.DataFrame({'R': real_vol, 'I': imp_vol})
    vol = df.max(axis=1)

    return vol


def x_tstats(x, decay):
    x2 = x * x
    beta = x.ewm(com=decay / (1. - decay), adjust=False).mean()
    gamma = x2.ewm(com=decay / (1. - decay), adjust=False).mean()
    window_size = round(1.5 * (1 + decay) / (1 - decay))
    std_err = np.sqrt((gamma - beta * beta) / (window_size - 1))
    tstat = beta / std_err
    return tstat, beta


def smooth_tstat(e, smt_param):
    if np.abs(e) > smt_param:
        return np.sign(e)
    else:
        return np.sin(e * np.pi / (2 * smt_param))


def hraw_replication(smt_tstats, spot, ir_X, ir_Y, modelVol):
    delta_shift = .4
    delta_atm = .5
    T = 4. / 52
    options = 5
    Delta4Ks = smt_tstats * delta_shift + delta_atm
    d1 = norm.ppf(Delta4Ks)
    cterm = (ir_Y - ir_X) * T / 100 + modelVol * modelVol * T / 2
    hraw = 0
    for i in range(1, options + 1):
        strike = spot.shift(i) / np.exp(np.sqrt(T) * d1 * modelVol.shift(i) - cterm.shift(i))
        delta = norm.cdf((np.log(spot / strike) + cterm) / (modelVol * np.sqrt(T)))
        hraw = hraw + delta
    hraw = hraw / options

    return hraw, Delta4Ks


def performance(signal, total_returns, tc=.0002):
    trans_cost = tc * (signal.diff()).abs()
    excess_return = total_returns * signal.shift(1) - trans_cost

    return excess_return


def get_signal(df, signal, decay=0.97, smt_param=1, tc=2e-4):
    results_df = pd.DataFrame(index=df.index)
    # Define variables from input data
    spot = df['Spot']
    ir_X = df['IR_X']
    ir_Y = df['IR_Y']
    imp_vol = df['ImpVol']
    riskrev_dif = df[signal].diff()

    spot_return = np.log(spot).diff()
    realVol = realised_vol(spot_return, decay)
    modelVol = model_vol(realVol, imp_vol)
    rr_tstat, beta = x_tstats(riskrev_dif, decay)
    spot_tstat, spot_momentum = x_tstats(spot_return, decay)
    smt_tstats = rr_tstat.apply(smooth_tstat, args=(smt_param,))
    hraw, Delta4Ks = hraw_replication(smt_tstats, spot, ir_X, ir_Y, modelVol)

    results_df['OR_fact'] = 2 * hraw - 1
    results_df['smt_tstats'] = smt_tstats

    pnl_OR_fact = performance(results_df['smt_tstats'], np.log(df['Spot']).diff(), tc=tc)

    results_df['PnL'] = pnl_OR_fact
    results_df['PnL_cum'] = pnl_OR_fact.cumsum()
    results_df[signal] = df[signal]
    results_df['Spot'] = df['Spot']

    return results_df

def get_data_for_OR(currency = 'JPYUSD'):

    # carry on long-term bond (difference between longer-term yields across currencies)
    df_ccy_10y_spread = get_ccy10y_spreads(currency=currency, daily_change=False)

    df_ccy_2y_spread = get_ccy2y_spreads(currency=currency, daily_change=False)

    # Business cycle factor (difference between the spread 2-10 years)
    df_spreads = relative_spread(currency=currency)
    df_spreads.index.name = 'Date'

    # commodity prices
    df_commodities = get_commodity_prices(daily_change=False)
    df_commodities['Copper_Gold_Ratio'] = df_commodities['Copper_adj_fut'] / df_commodities['Gold_adj_fut']

    df = df_ccy_10y_spread.reset_index().merge(df_ccy_2y_spread.reset_index())
    df = df.merge(df_spreads.reset_index())
    df = df.merge(df_commodities.reset_index())
    df = df.set_index('Date')

    # add currency returns
    # load files
    df_ccy = get_data(currency=currency, get_transformed_data=False)
    df_ccy = df_ccy[df_ccy['Currency'] == currency]

    df = df_ccy.reset_index().merge(df.reset_index())
    return df.set_index('Date')
