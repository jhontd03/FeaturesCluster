import pandas_ta as ta
import pandas as pd

import indicators as ind


def extract_features_basic(data_ohlc: pd.DataFrame):
    
    """
    Extrae un conjunto de características técnicas de una serie de datos 
    OHLC (Open, High, Low, Close).

    Parámetros:
    -----------
    data_ohlc : pd.DataFrame
        DataFrame que contiene columnas 'open', 'high', 'low' y 'close' 
        con los datos OHLC.

    Retorna:
    --------
    data : pd.DataFrame
        DataFrame con las características calculadas. Las columnas incluyen:

        - 'adx_7', 'adx_14', 'adx_21', 'adx_50', 'adx_75', 'adx_100': Índice de 
           Movimiento Direccional Promedio (ADX) para diferentes longitudes.
        - 'plusdi_7', 'plusdi_14', 'plusdi_21': Indicador Direccional Positivo 
          (+DI) para diferentes longitudes.
        - 'minusdi_7', 'minusdi_14', 'minusdi_21': Indicador Direccional 
           Negativo (-DI) para diferentes longitudes.
        - 'cbbu_14_2', 'cbbl_14_2', 'cbbu_20_2', 'cbbl_20_2': 
           Precio de cierre normalizado por las Bandas de Bollinger superiores 
           e inferiores para diferentes longitudes.
        - 'cma_7', 'cma_14', 'cma_21', 'cma_28', 'cma_50', 'cma_100', 'cma_200': 
           Precio de cierre normalizado por la Media Móvil Simple (SMA) 
           para diferentes longitudes.
        - 'atratr_7_7', 'atratr_14_14', 'atratr_21_21': ATR normalizado por su 
           propio valor desplazado para diferentes longitudes.
        - 'cci_7', 'cci_14', 'cci_21', 'cci_28': Índice de Canal de Materias 
           Primas (CCI) para diferentes longitudes.
        - 'mom_7', 'mom_14', 'mom_21', 'mom_28': Momentum para diferentes
          longitudes.
        - 'rsi_2', 'rsi_7', 'rsi_14': Índice de Fuerza Relativa (RSI) 
           para diferentes longitudes.
        - 'wpr_7', 'wpr_14', 'wpr_21': Oscilador de Williams (%R) 
           para diferentes longitudes.
        - 'stoch_5_3_3', 'stoch_14_3_3', 'stoch_20_5_5': Oscilador Estocástico
           %D para diferentes longitudes.
        - 'cc_1', 'cc_2', 'cc_3', 'cc_60', 'cc_30', 'cc_15': Cambio porcentual 
           del precio de cierre para diferentes desplazamientos.
        - 'll_1', 'll_2', 'll_3': Cambio porcentual del precio mínimo para 
           diferentes desplazamientos.
        - 'hh_1', 'hh_2', 'hh_3': Cambio porcentual del precio máximo para 
           diferentes desplazamientos.
        - 'hc': Relación del precio máximo al precio de cierre.
        - 'cl': Relación del precio de cierre al precio mínimo.
        - 'atr1atr_200': Relación de ATR de longitud 1 con ATR de longitud 200.
        - 'dma_7', 'dma_21', 'dma_50': Relación de la SMA con su valor 
           desplazado para diferentes longitudes.
        - 'dm_10', 'dm_40', 'dm_80': Demarker para diferentes longitudes.
        - 'rvi_10', 'rvi_25', 'rvi_50', 'rvi_75', 'rvi_100', 'rvi_150': 
           Índice de Volatilidad Relativa (RVI) para diferentes longitudes.

    Notas:
    ------
    - Esta función utiliza una variedad de indicadores técnicos para 
      proporcionar una amplia gama de características que pueden ser utilizadas 
      en análisis o modelos predictivos.
    - Se realiza un desplazamiento en los datos de entrada para evitar el uso 
      de datos futuros en el cálculo de las características.
    - Se eliminan las filas con valores NaN resultantes de los cálculos de 
      los indicadores.
    """    

    data = pd.DataFrame(index=data_ohlc.index)

    # Se crea shift para evitar filtrar datos de futuro
    dopen = data_ohlc.open.shift(1)
    dhigh = data_ohlc.high.shift(1)
    dlow = data_ohlc.low.shift(1)
    dclose = data_ohlc.close.shift(1)
    dopen.dropna(inplace=True)
    dhigh.dropna(inplace=True)
    dlow.dropna(inplace=True)
    dclose.dropna(inplace=True)
    
    # data['hour'] = pd.Series(dclose.index.hour, index=dclose.index)

    adx_7 = ind.adx(dhigh, dlow, dclose, length=7)
    adx_14 = ind.adx(dhigh, dlow, dclose, length=14)
    adx_21 = ind.adx(dhigh, dlow, dclose, length=21)

    data['adx_7'] = adx_7.iloc[:, 0]
    data['adx_14'] = adx_14.iloc[:, 0]
    data['adx_21'] = adx_21.iloc[:, 0]

    adx_50 = ind.adx(dhigh, dlow, dclose, length=50)
    adx_75 = ind.adx(dhigh, dlow, dclose, length=75)
    adx_100 = ind.adx(dhigh, dlow, dclose, length=100)

    data['adx_50'] = adx_50.iloc[:, 0]
    data['adx_75'] = adx_75.iloc[:, 0]
    data['adx_100'] = adx_100.iloc[:, 0]

    data['plusdi_7'] = adx_7.iloc[:, 2]
    data['plusdi_14'] = adx_14.iloc[:, 2]
    data['plusdi_21'] = adx_21.iloc[:, 2]

    data['minusdi_7'] = adx_7.iloc[:, 3]
    data['minusdi_14'] = adx_14.iloc[:, 3]
    data['minusdi_21'] = adx_21.iloc[:, 3]

    data['cbbu_14_2'] = dclose/ta.bbands(dclose, length=14, std=2).loc[:, 'BBU_14_2.0']
    data['cbbl_14_2'] = dclose/ta.bbands(dclose, length=14, std=2).loc[:, 'BBL_14_2.0']
    data['cbbu_20_2'] = dclose/ta.bbands(dclose, length=20, std=2).loc[:, 'BBU_20_2.0']
    data['cbbl_20_2'] = dclose/ta.bbands(dclose, length=20, std=2).loc[:, 'BBL_20_2.0']

    data['cma_7'] = dclose/ta.sma(close=dclose, length=7)
    data['cma_14'] = dclose/ta.sma(close=dclose, length=14)
    data['cma_21'] = dclose/ta.sma(close=dclose, length=21)
    data['cma_28'] = dclose/ta.sma(close=dclose, length=28)
    data['cma_50'] = dclose/ta.sma(close=dclose, length=50)
    data['cma_100'] = dclose/ta.sma(close=dclose, length=100)
    data['cma_200'] = dclose/ta.sma(close=dclose, length=200)

    atr7 = ind.atr(dhigh, dlow, dclose, length=7) 
    atr14 = ind.atr(dhigh, dlow, dclose, length=14) 
    atr21 = ind.atr(dhigh, dlow, dclose, length=21) 

    data['atratr_7_7'] = atr7/atr7.shift(7)
    data['atratr_14_14'] = atr14/atr14.shift(14)
    data['atratr_21_21'] = atr21/atr21.shift(21)

    data['cci_7'] = ta.cci(dclose, dclose, dclose, length=7)
    data['cci_14'] = ta.cci(dclose, dclose, dclose, length=14)
    data['cci_21'] = ta.cci(dclose, dclose, dclose, length=21)
    data['cci_28'] = ta.cci(dclose, dclose, dclose, length=28)

    data['mom_7'] = ind.momentum(dclose, length=7)
    data['mom_14'] = ind.momentum(dclose, length=14)
    data['mom_21'] = ind.momentum(dclose, length=21)
    data['mom_28'] = ind.momentum(dclose, length=28)

    data['rsi_2'] = ta.rsi(close=dclose, length=2)
    data['rsi_7'] = ta.rsi(close=dclose, length=7)
    data['rsi_14'] = ta.rsi(close=dclose, length=14)

    data['wpr_7'] = ta.willr(dhigh, dlow, dclose, length=7)
    data['wpr_14'] = ta.willr(dhigh, dlow, dclose, length=14)
    data['wpr_21'] = ta.willr(dhigh, dlow, dclose, length=21)

    data['stoch_5_3_3'] = ta.stoch(dhigh, dlow, dclose, k=5, d=3, smooth_k=3, mamode='sma',).iloc[:, 1]
    data['stoch_14_3_3'] = ta.stoch(dhigh, dlow, dclose, k=14, d=3, smooth_k=3, mamode='sma',).iloc[:, 1]
    data['stoch_20_5_5'] = ta.stoch(dhigh, dlow, dclose, k=20, d=5, smooth_k=5, mamode='sma',).iloc[:, 1]

    data['cc_1'] = dclose/dclose.shift(1)
    data['cc_2'] = dclose/dclose.shift(2)
    data['cc_3'] = dclose/dclose.shift(3)

    data['ll_1'] = dlow/dlow.shift(1)
    data['ll_2'] = dlow/dlow.shift(2)
    data['ll_3'] = dlow/dlow.shift(3)

    data['hh_1'] = dhigh/dhigh.shift(1)
    data['hh_2'] = dhigh/dhigh.shift(2)
    data['hh_3'] = dhigh/dhigh.shift(3)

    data['cc_60'] = dclose/dclose.shift(60)
    data['cc_30'] = dclose/dclose.shift(30)
    data['cc_15'] = dclose/dclose.shift(15)

    data['hc'] = dhigh/dclose
    data['cl'] = dclose/dlow

    data['atr1atr_200'] = ind.atr(dhigh, dlow, dclose, length=1) / \
        ind.atr(dhigh, dlow, dclose, length=200)

    data['dma_7'] = ta.sma(close=dclose, length=7) / \
        ta.sma(close=dclose, length=7, offset=7)
    data['dma_21'] = ta.sma(close=dclose, length=21) / \
        ta.sma(close=dclose, length=21, offset=21)
    data['dma_50'] = ta.sma(close=dclose, length=50) / \
        ta.sma(close=dclose, length=50, offset=50)

    data['dm_10'] = ind.demarker(dhigh, dlow, dclose, length=10)
    data['dm_40'] = ind.demarker(dhigh, dlow, dclose, length=40)
    data['dm_80'] = ind.demarker(dhigh, dlow, dclose, length=80)

    data['rvi_10'] = ta.rvgi(dopen, dhigh, dlow, dclose, length=10).loc[:, 'RVGI_10_4']
    data['rvi_25'] = ta.rvgi(dopen, dhigh, dlow, dclose, length=25).loc[:, 'RVGI_25_4']
    data['rvi_50'] = ta.rvgi(dopen, dhigh, dlow, dclose, length=50).loc[:, 'RVGI_50_4']
    data['rvi_75'] = ta.rvgi(dopen, dhigh, dlow, dclose, length=75).loc[:, 'RVGI_75_4']
    data['rvi_100'] = ta.rvgi(dopen, dhigh, dlow, dclose, length=100).loc[:, 'RVGI_100_4']
    data['rvi_150'] = ta.rvgi(dopen, dhigh, dlow, dclose, length=150).loc[:, 'RVGI_150_4']

    data.dropna(inplace=True)
    
    return data