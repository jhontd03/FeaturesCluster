"""
data_labeling.py
---------------
This script contains functions for processing and labeling financial market 
data in OHLC (Open, High, Low, Close) format. Specifically, it includes 
the `raw_returns_labeling` function, which calculates and labels raw 
returns based on a specified bias.

Author: [Your Name]
Date: [Date]
Version: 1.0

Usage:
------
This script can be imported as a module in other Python scripts, 
or executed directly to process market data. Make sure to install 
the required dependencies, such as `pandas` and `numpy`, before running.

Dependencies:
-------------
- pandas
- numpy

Functions:
----------
- raw_returns_labeling: Labels raw returns based on a specified bias.
"""

import numpy as np
import pandas as pd

def raw_returns_labeling(data_ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Label the raw returns of stock market data based on specified bias.

    This function takes a DataFrame containing OHLC (Open, High, Low, Close) 
    data and labels the returns based on a bias and a specified shift.

    Parameters
    ----------
    data_ohlc : pd.DataFrame
        A DataFrame containing stock market OHLC data with at least an 'open' column.
    
    **kwargs : keyword arguments
        - bias_extraction : bool
            If True, applies bias extraction to compute returns.
        - type_bias : {'up', 'down'}
            Specifies the type of bias to extract. 'up' for upward bias and 'down' 
            for downward bias.
        - bias : float
            The threshold for labeling returns as 'UP' or 'DOWN'. This is only used 
            if bias_extraction is True.
        - shift : int
            The number of periods to shift for calculating returns. Determines how far 
            into the future the function looks to label the returns.

    Returns
    -------
    pd.DataFrame
        A DataFrame with an additional column 'LABEL' that categorizes each open price 
        movement as 'UP' or 'DOWN' based on the specified criteria.

    Notes
    -----
    The function modifies the DataFrame in place by adding the 'LABEL' column. 
    Ensure that the DataFrame passed has the required 'open' column and is properly 
    formatted.
    
    Examples
    --------
    >>> df = pd.DataFrame({'open': [100, 102, 101, 104]})
    >>> labeled_df = raw_returns_labeling(df, bias_extraction=True, type_bias='up', bias=0.01, shift=1)
    >>> print(labeled_df)
    """

    data = data_ohlc.copy(deep=True)

    if kwargs['bias_extraction']:
        if kwargs['type_bias'] == 'up':
            returns = data.open.shift(-kwargs['shift']) / data.open
            data['LABEL'] = np.where((returns >= 1 + kwargs['bias']), 'UP', 'DOWN')
        elif kwargs['type_bias'] == 'down':
            returns = data.open / data.open.shift(-kwargs['shift'])
            data['LABEL'] = np.where((returns >= 1 + kwargs['bias']), 'DOWN', 'UP')
    else:
        if kwargs['type_bias'] == 'up':
            data['LABEL'] = np.where(data.open.shift(-kwargs['shift']) > data.open, 'UP', 'DOWN')
        elif kwargs['type_bias'] == 'down':
            data['LABEL'] = np.where(data.open > data.open.shift(-kwargs['shift']), 'DOWN', 'UP')

    return data
