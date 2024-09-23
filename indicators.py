import pandas as pd
import numpy as np
from pandas_ta.utils import zero
from pandas_ta.overlap import ma

def adx(high, low, close, length):

    data = pd.DataFrame()

    data['pos'] = high - high.shift(1)
    data['neg'] = low.shift(1) - low
    data['pos'] = ((data['pos'] > data['neg']) & (data['pos'] > 0)) * data['pos']
    data['neg'] = ((data['neg'] > data['pos']) & (data['neg'] > 0)) * data['neg']
    data['pos'] = data['pos'].apply(zero)
    data['neg'] = data['neg'].apply(zero)

    data['tr'] = tr(high, low, close)

    data['pd'] = data['pos']/data['tr']
    data['md'] = data['neg']/data['tr']

    data.fillna(0, inplace=True)

    data[f'pdi_{length}'] = 100*ma('ema', data['pd'], length=length)
    data[f'mdi_{length}'] = 100*ma('ema', data['md'], length=length)

    data[f'dx_{length}'] = 100*abs(data[f'pdi_{length}'] - data[f'mdi_{length}']) / \
                                  (data[f'pdi_{length}'] + data[f'mdi_{length}'])

    data[f'adx_{length}'] = ma('ema', data[f'dx_{length}'], length=length)

    return data.loc[:, [f'adx_{length}', f'dx_{length}', f'pdi_{length}', f'mdi_{length}']]


def atr(high, low, close, length):
    tr_ = tr(high, low, close)
    atr = tr_.rolling(length).sum()/length
    atr.name = f'atr_{length}'
    return atr

def tr(high, low, close):
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    tr = np.max(ranges, axis=1)
    return tr

def momentum(close, length):
    mom = 100*close/close.shift(length)
    mom.name = f'mom_{length}'
    return mom

def demarker(high, low, close, length=14):

    data = pd.DataFrame()
    data['hdif'] = high > high.shift(1)
    data['hsub'] = high - high.shift(1)
    data['demax'] = np.where(data['hdif'] == False, 0, data['hsub'])

    data['ldif'] = low < low.shift(1)
    data['lsub'] = low.shift(1) - low
    data['demin'] = np.where(data['ldif'] == False, 0, data['lsub'])

    data["sma_demax"] = data["demax"].rolling(window=length).mean()
    data["sma_demin"] = data["demin"].rolling(window=length).mean()

    result = data["sma_demax"] / (data["sma_demax"] + data["sma_demin"])

    result.name = f'dma_{length}'

    return result