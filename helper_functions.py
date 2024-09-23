import pandas as pd
import numpy as np
import  datetime as dt

from numba import jit
import seaborn as sns

from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import MetaTrader5 as mt5
from nancorrmp import NaNCorrMp
import matplotlib.pyplot as plt


def plot_kde(X_train, X_test, feature):
    """
    Esta función grafica la estimación de densidad de kernel (KDE)
    para la característica especificada en dos conjuntos de datos: X_train y X_test.

    Parámetros:
    X_train : DataFrame
        Conjunto de datos de entrenamiento.
    X_test : DataFrame
        Conjunto de datos de prueba.
    feature : str
        Nombre de la característica a graficar.
    """
    # Verifica que la característica esté en ambos DataFrames
    if feature not in X_train.columns or feature not in X_test.columns:
        raise ValueError(f"La característica '{feature}' no se encuentra en uno de los DataFrames.")
    
    # Graficar la densidad para X_train con un color específico
    sns.kdeplot(X_train[feature], shade=True, label="X_train", color='blue', alpha=0.5)
    
    # Graficar la densidad para X_test con un color diferente
    sns.kdeplot(X_test[feature], shade=True, label="X_test", color='orange', alpha=0.5)
    
    # Configurar los ticks de los ejes
    plt.xticks(rotation=45)
    
    # Eliminar el borde izquierdo de la gráfica
    sns.despine(left=True)
    
    # Agregar leyenda
    plt.legend(title=f"{feature}")

    # Mostrar la gráfica
    plt.title(f'Distribution of {feature}')
    plt.show()



def split_dates(start_date: str, end_date: str, proportion=0.7):
    """Divide un rango de fechas en una proporción determinada.

    Esta función toma una fecha de inicio y una fecha de fin, genera un rango de fechas 
    entre estas dos y luego devuelve la fecha correspondiente al final de la parte 
    proporcional del rango.

    Parameters:
        start_date (str): La fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): La fecha de fin en formato 'YYYY-MM-DD'.
        proportion (float, opcional): La proporción del rango de fechas que se desea 
            devolver. Por defecto es 0.7, lo que significa que se devuelve el 70% 
            de las fechas. Debe estar entre 0 y 1.

    Returns:
        str: La fecha en formato 'YYYY-MM-DD' correspondiente al final de la parte 
        proporcional del rango de fechas.

    Ejemplo:
        >>> split_dates("2022-01-01", "2022-01-10", 0.7)
        '2022-01-07'
    """
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    dates = pd.date_range(start=start_date, end=end_date)
    indexs = int(len(dates) * proportion)
    oos_date = dt.datetime.strftime(dates[:indexs][-1], "%Y-%m-%d")

    return oos_date



def data_import_MT5(symbol: str,
                    start: str,
                    end: str = dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d"),
                    time_frame: str = mt5.TIMEFRAME_D1) -> pd.DataFrame:
    """
    Importa datos OHLCV de MetaTrader 5 para un símbolo dado y un rango 
    fechas especificado.

    Parámetros:
    -----------
    symbol : str
        El símbolo del activo financiero a importar.
    
    start : str
        Fecha de inicio en formato 'YYYY-MM-DD'.
    
    end : str, opcional
        Fecha de fin en formato 'YYYY-MM-DD' (por defecto es la fecha actual).
    
    time_frame : str, opcional
        Marco de tiempo de los datos a importar (por defecto es diario - mt5.TIMEFRAME_D1).
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame que contiene los datos OHLCV 
        ('open', 'high', 'low', 'close', 'spread') con el índice de fechas.
    
    Notas:
    ------
    - La función inicializa la conexión con MetaTrader 5 y verifica 
      que la inicialización sea exitosa.
    - Convierte las fechas de inicio y fin a UTC.
    - Utiliza `mt5.copy_rates_range` para copiar los datos dentro del rango 
      de fechas especificado.
    - Convierte los datos a un DataFrame de pandas y ajusta el índice de fecha.
    - Apaga la conexión con MetaTrader 5 al finalizar.
    - Si la inicialización de MetaTrader 5 falla, imprime un mensaje 
      de error y apaga la conexión.
    """
    
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()

    start = dt.datetime.strptime(start, '%Y-%m-%d')
    start_utc = start.astimezone(dt.timezone.utc)

    end = dt.datetime.strptime(end, '%Y-%m-%d')
    end_utc = end.astimezone(dt.timezone.utc)

    rates = mt5.copy_rates_range(symbol,
                                 time_frame,
                                 start_utc,
                                 end_utc)

    data_ohlcv = pd.DataFrame(rates)

    data_ohlcv['date'] = pd.to_datetime(data_ohlcv['time'], unit='s')
    data_ohlcv = data_ohlcv.set_index(data_ohlcv['date'])
    data_ohlcv = data_ohlcv.loc[:, ['open', 'high', 'low', 'close', 'spread']]

    mt5.shutdown()

    return data_ohlcv
   

def graphs_quantile_correlation(X: pd.Series, y: pd.Series, quantiles: int=10, shift: int=None):
    """
    Genera gráficos para visualizar la correlación entre los deciles 
    de una característica y los rendimientos.

    Parámetros:
    -----------
    X : pd.Series
        Serie que contiene la característica a analizar.
    
    y : pd.Series
        Serie que contiene los rendimientos asociados.
    
    shift : int
        Desplazamiento de días para la predicción.

    Retorna:
    --------
    None
        La función genera y muestra gráficos, pero no retorna ningún valor.
    
    Notas:
    ------
    - La función crea un DataFrame combinado de X y y, y calcula los deciles 
      para ambos.
    - Calcula los rendimientos promedio por decil de la característica.
    - Genera un histograma de la distribución de la característica y un gráfico
      de barras de los rendimientos por decil.
    - Llama a la función `scatter_corr` para generar un gráfico de dispersión 
      con la correlación.
    """
    feature = X.name  

    quantile_df = pd.concat([X, y], axis=1).dropna()
    quantile_df['feature_deciles'] = \
        pd.qcut(quantile_df[feature], q=quantiles, precision=0, duplicates='drop')
    quantile_df['returns_deciles'] = \
        pd.qcut(quantile_df['returns'], q=quantiles, precision=0, duplicates='drop')

    rpc = quantile_df.groupby(by='feature_deciles', observed=True)['returns'].mean()

    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=[f'{feature} Distribution', 'Returns per Quantile'])
    fig.add_trace(go.Histogram(x=X, nbinsx=100), row=1, col=1)
    fig.add_trace(go.Bar(x=rpc.index.astype(str), y=rpc), row=2, col=1)
    fig.update_layout(
        height=700, 
        template='plotly_dark', 
        title_text=f"{feature} - {shift} bars prediction")
    fig.update_xaxes(title_text='Feature Quantiles', row=2, col=1)
    fig.update_yaxes(title_text='Returns', row=2, col=1)
    plot(fig, auto_open=True, filename="quantile_ind_returns.html")
    
    return


def drop_corr_features_target(data_features: pd.DataFrame,
                              data_corr: pd.Series,
                              threshold_corr: int = 0.7,
                              n_cpus: int = 4) -> list:
    
    """
    Elimina características altamente correlacionadas de un DataFrame 
    de características basado en la correlación con el objetivo 
    y la correlación media de las características.

    Parámetros:
    -----------
    data_features : pd.DataFrame
        DataFrame que contiene las características a analizar.
    
    data_corr : pd.Series
        Serie que contiene la correlación de cada característica con el objetivo.
    
    threshold_corr : float, opcional
        Umbral de correlación para considerar características como altamente 
        correlacionadas (por defecto es 0.7).
    
    n_cpus : int, opcional
        Número de CPUs a utilizar para el cálculo paralelo de la matriz de 
        correlación (por defecto es 4).
    
    Retorna:
    --------
    list
        Lista de nombres de las características que se deben eliminar 
        debido a la alta correlación.
    
    Notas:
    ------
    - La función utiliza `NaNCorrMp.calculate` para calcular la matriz 
    de correlación, eliminando filas con valores NaN en el proceso.
    - Se calcula la matriz de correlación absoluta y se obtienen l
    as correlaciones superiores triangulares para evitar duplicados.
    - Las características a eliminar se determinan mediante una comparación 
    de correlación con el objetivo y la correlación media de las características.
    """    
    corr_mtx = abs(NaNCorrMp.calculate(data_features, n_cpus))
    avg_corr = corr_mtx.mean(axis=1)
    upper_tri = corr_mtx.where(
        np.triu(np.ones(corr_mtx.shape), k=1).astype(bool)
        )
    
    upper_tri_matrix = upper_tri.to_numpy()
    data_corr_values = data_corr.to_numpy()
    avg_corr_values = avg_corr.to_numpy()
    features = np.array(upper_tri.columns.to_list())
    
    features_to_drop = drop_features_numba(upper_tri_matrix,
                                           data_corr_values,
                                           avg_corr_values,
                                           features, 
                                           threshold_corr)

    dropcols_names = list(set(features_to_drop))

    return dropcols_names

@jit(nopython=True)
def drop_features_numba(upper_tri_matrix: np.array,
                        data_corr_values: np.array,
                        avg_corr_values: np.array,
                        features: np.array,
                        threshold_corr: int):
    
    """
    Función auxiliar que determina qué características eliminar basándose 
    en la matriz de correlación superior, las correlaciones 
    con el objetivo y las correlaciones medias.

    Parámetros:
    -----------
    upper_tri_matrix : np.array
        Matriz triangular superior de correlaciones entre características.
    
    data_corr_values : np.array
        Array de correlaciones de cada característica con el objetivo.
    
    avg_corr_values : np.array
        Array de correlaciones medias de cada característica.
    
    features : np.array
        Array de nombres de las características.
    
    threshold_corr : float
        Umbral de correlación para considerar características como altamente 
        correlacionadas.
    
    Retorna:
    --------
    list
        Lista de nombres de las características que se deben eliminar 
        debido a la alta correlación.
    """    

    features_to_drop = []
    for row in range(len(upper_tri_matrix) - 1):
        for col in range(row + 1, len(upper_tri_matrix)):
            if upper_tri_matrix[row, col] > threshold_corr:

                if data_corr_values[row] < data_corr_values[col]:
                    features_to_drop.append(features[row])
                elif data_corr_values[row] > data_corr_values[col]:
                    features_to_drop.append(features[col])
                else:
                    if avg_corr_values[row] > avg_corr_values[col]:
                        features_to_drop.append(features[col])
                    else:
                        features_to_drop.append(features[row])

    return features_to_drop
