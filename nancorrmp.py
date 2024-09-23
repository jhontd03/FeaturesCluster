"""
This module contains the NaNCorrMp class, which provides methods for computing
correlation matrices in parallel while handling NaN values effectively. It is designed
for use with large datasets and multi-core architectures.

This original module was developed by: https://github.com/bukson/nancorrmp
"""
import ctypes
from multiprocessing.sharedctypes import RawArray
import multiprocessing as mp
from typing import Tuple, Union
import numpy as np
import pandas as pd
from scipy.special import btdtr

shared_variables_dictionary = {}
ArrayLike = Union[pd.DataFrame, np.ndarray]

class NaNCorrMp(object):
    """
    A class for parallel computation of correlation matrices while
    managing NaN values in input data arrays.

    Methods
    -------
    calculate(X: ArrayLike, n_jobs: int = -1, chunks: int = 500) -> ArrayLike
        Calculate the correlation matrix without p-values.
    calculate_with_p_value(X: ArrayLike, n_jobs: int = -1, chunks: int = 500) -> Tuple[ArrayLike, ArrayLike]
        Calculate the correlation matrix with p-values.
    """
    @staticmethod
    def _init_worker(X: RawArray, X_finite_mask: RawArray, X_corr: RawArray,
                     X_shape: Tuple[int, int], X_corr_shape: Tuple[int, int],
                     X_p_value: RawArray = None) -> None:
        """
        Initialize worker function.

        Parameters
        ----------
        X : RawArray
            Shared data array.
        X_finite_mask : RawArray
            Mask for finite values in X.
        X_corr : RawArray
            Array to store correlation results.
        X_shape : Tuple[int, int]
            Shape of the input data array.
        X_corr_shape : Tuple[int, int]
            Shape of the correlation array.
        X_p_value : RawArray, optional
            (default is None) Array to store p-values (if any).

        Returns
        -------
        None
        """
        shared_variables_dictionary['X'] = X
        shared_variables_dictionary['X_finite_mask'] = X_finite_mask
        shared_variables_dictionary['X_corr'] = X_corr
        shared_variables_dictionary['X_shape'] = X_shape
        shared_variables_dictionary['X_corr_shape'] = X_corr_shape
        shared_variables_dictionary['X_p_value'] = X_p_value
        shared_variables_dictionary['X_p_value_shape'] = X_corr_shape

    @staticmethod
    def calculate(X: ArrayLike, n_jobs: int = -1, chunks: int = 500) -> ArrayLike:
        """
        Calculate the correlation matrix for the input data.

        Parameters
        ----------
        X : ArrayLike
            Input data as a DataFrame or ndarray.
        n_jobs : int, optional
            Number of jobs to run in parallel (default is -1, using all available cores).
        chunks : int, optional
            Size of the chunks to split the tasks into (default is 500).

        Returns
        -------
        ArrayLike
            Correlation matrix as a DataFrame or ndarray.
        """
        return NaNCorrMp._calculate(X=X, n_jobs=n_jobs, chunks=chunks, add_p_values=False)

    @staticmethod
    def calculate_with_p_value(X: ArrayLike,
                               n_jobs: int = -1,
                               chunks: int = 500
                               ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Calculate the correlation matrix with p-values for the input data.

        Parameters
        ----------
        X : ArrayLike
            Input data as a DataFrame or ndarray.
        n_jobs : int, optional
            Number of jobs to run in parallel (default is -1, using all available cores).
        chunks : int, optional
            Size of the chunks to split the tasks into (default is 500).

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            Correlation matrix and the corresponding p-values as DataFrame or ndarray.
        """
        return NaNCorrMp._calculate(X=X, n_jobs=n_jobs, chunks=chunks, add_p_values=True)

    @staticmethod
    def _calculate(X: ArrayLike,
                   n_jobs: int,
                   chunks: int,
                   add_p_values: int
                   ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        """
        Internal method to calculate the correlation matrix and p-values.

        Parameters
        ----------
        X : ArrayLike
            Input data as a DataFrame or ndarray.
        n_jobs : int
            Number of jobs to run in parallel.
        chunks : int
            Size of the chunks to split the tasks into.
        add_p_values : int
            A flag indicating whether to compute p-values (1) or not (0).

        Returns
        -------
        Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]
            Correlation matrix as DataFrame or ndarray and optionally the p-values.
        """   
        X_array = X.to_numpy(dtype=np.float64, copy=True).transpose() if type(X) == pd.DataFrame else X
        X_raw = RawArray(ctypes.c_double, X_array.shape[0] * X_array.shape[1])
        X_np = np.frombuffer(X_raw, dtype=np.float64).reshape(X_array.shape)
        np.copyto(X_np, X_array)

        finite_mask_data = np.isfinite(X_array)
        finite_mask_raw = RawArray(ctypes.c_bool, X_array.shape[0] * X_array.shape[1])
        finite_mask_np = np.frombuffer(finite_mask_raw, dtype=np.bool_).reshape(X_array.shape)
        np.copyto(finite_mask_np, finite_mask_data)

        X_corr = np.ndarray(shape=(X_array.shape[0], X_array.shape[0]), dtype=np.float64)
        X_corr_raw = RawArray(ctypes.c_double, X_corr.shape[0] * X_corr.shape[1])
        X_corr_np = np.frombuffer(X_corr_raw, dtype=np.float64).reshape(X_corr.shape)

        if add_p_values:
            X_p_value = np.ndarray(shape=X_corr.shape, dtype=np.float64)
            X_p_value_raw = RawArray(ctypes.c_double, X_p_value.shape[0] * X_p_value.shape[1])
            X_p_value_np = np.frombuffer(X_p_value_raw, dtype=np.float64).reshape(X_corr.shape)
        else:
            X_p_value_np = None
            X_p_value_raw = None
            X_p_value_np = None

        arguments = ((j, i) for i in range(X_array.shape[0]) for j in range(i))
        processes = n_jobs if n_jobs > 0 else None
        worker_function = NaNCorrMp._set_correlation_with_p_value if add_p_values else NaNCorrMp._set_correlation
        with mp.Pool(processes=processes,
                     initializer=NaNCorrMp._init_worker,
                     initargs=(X_raw, finite_mask_raw, X_corr_raw, X_np.shape, X_corr_np.shape, X_p_value_raw)) \
                as pool:
            list(pool.imap_unordered(worker_function, arguments, chunks))

        for i in range(X_corr_np.shape[0]):
            X_corr_np[i][i] = 1.0

        if add_p_values:
            if type(X) == pd.DataFrame:
                return (
                    pd.DataFrame(X_corr_np, columns=X.columns, index=X.columns),
                    pd.DataFrame(X_p_value_np, columns=X.columns, index=X.columns)
                )
            else:
                return X_corr_np, X_p_value_np

        if type(X) == pd.DataFrame:
            return pd.DataFrame(X_corr_np, columns=X.columns, index=X.columns)
        else:
            return X_corr_np

    @staticmethod
    def _set_correlation(arguments: Tuple[int, int]) -> None:
        """
        Set correlation values.

        Parameters
        ----------
        arguments : Tuple[int, int]
            Indices of the two variables to correlate.

        Returns
        -------
        None
        """
        j, i = arguments
        X_np, X_corr_np, finite_mask = NaNCorrMp._get_global_variables()
        finites = finite_mask[i] & finite_mask[j]
        x = X_np[i][finites]
        y = X_np[j][finites]
        corr = NaNCorrMp._corr(x, y)
        X_corr_np[i][j] = corr
        X_corr_np[j][i] = corr

    @staticmethod
    def _get_global_variables(get_p_value: bool = False) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                                  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Retrieve shared global variables.

        Parameters
        ----------
        get_p_value : bool, optional
            Flag to determine if p-values should be retrieved (default is False).

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
              Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            Global variables including data array, correlation array and mask, and optionally p-values.
        """
        X_np = np.frombuffer(
            shared_variables_dictionary['X'],
            dtype=np.float64).reshape(shared_variables_dictionary['X_shape']
            )
        X_corr_np = np.frombuffer(
            shared_variables_dictionary['X_corr'],
            dtype=np.float64).reshape(shared_variables_dictionary['X_corr_shape']
            )
        finite_mask = np.frombuffer(
            shared_variables_dictionary['X_finite_mask'],
            dtype=bool).reshape(shared_variables_dictionary['X_shape']
            )
        if not get_p_value:
            return X_np, X_corr_np, finite_mask
        else:
            X_p_value_np = np.frombuffer(
                shared_variables_dictionary['X_p_value'],
                dtype=np.float64).reshape(shared_variables_dictionary['X_p_value_shape']
                                          )
            return X_np, X_corr_np, finite_mask, X_p_value_np

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the correlation between two numpy arrays.

        Parameters
        ----------
        x : np.ndarray
            First array.
        y : np.ndarray
            Second array.

        Returns
        -------
        float
            Correlation coefficient between x and y.
        """
        mx, my = x.mean(), y.mean()
        xm, ym = x - mx, y - my
        r_num = np.add.reduce(xm * ym)
        r_den = np.sqrt((xm * xm).sum() * (ym * ym).sum())
        r = r_num / r_den
        return max(min(r, 1.0), -1.0)

    @staticmethod
    def _set_correlation_with_p_value(arguments: Tuple[int, int]) -> None:
        """
        Set correlation and p-value results.

        Parameters
        ----------
        arguments : Tuple[int, int]
            Indices of the two variables to correlate.

        Returns
        -------
        None
        """
        j, i = arguments
        X_np, X_corr_np, finite_mask, X_p_value_np = \
            NaNCorrMp._get_global_variables(get_p_value=True)
        finites = finite_mask[i] & finite_mask[j]
        x = X_np[i][finites]
        y = X_np[j][finites]
        corr = NaNCorrMp._corr(x, y)
        X_corr_np[i][j] = corr
        X_corr_np[j][i] = corr
        p_value = NaNCorrMp._p_value(corr, len(x))
        X_p_value_np[i][j] = p_value
        X_p_value_np[j][i] = p_value

    @staticmethod
    def _p_value(corr: float, observation_length: int) -> float:
        """
        Calculate the p-value associated with a correlation coefficient.

        Parameters
        ----------
        corr : float
            Correlation coefficient.
        observation_length : int
            Number of observations.

        Returns
        -------
        float
            The calculated p-value.
        """
        ab = observation_length / 2 - 1
        if ab == 0:
            p_value = 1.0
        else:
            p_value = 2 * btdtr(ab, ab, 0.5 * (1 - abs(np.float64(corr))))
        return p_value
