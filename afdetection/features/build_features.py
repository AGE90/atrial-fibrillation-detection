import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple
from sklearn.impute import SimpleImputer

from sklearn import set_config
set_config(transform_output = "pandas")


class BuildFeatures:
    
   
    def features_target_split(
        self, 
        dataset: pd.DataFrame, 
        drop_cols: list, 
        target: str
        ) -> Tuple[pd.DataFrame, pd.Series]:
        """_summary_
        Parameters
        ----------
        dataset : pd.DataFrame
            _description_
        drop_cols : list
            _description_
        target : str
            _description_
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            _description_
        """
        
        # Dictionary to hold values for ritmi column (encoding)
        num_di = {'SR': 0, 'AF': 1, 'VA': 2}
        dataset = dataset.replace({target: num_di})
        
        X = dataset.drop(drop_cols, axis=1)
        X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)  
        y = dataset[target]
        return X, y
    
    def dominant_frequency(
        self,
        x: np.ndarray, 
        fs: float, 
        method: str = 'fft', 
        nperseg: int = None, 
    ) -> float:
        """Computes the dominant frequency of a signal

        Parameters
        ----------
        x : np.ndarray
            Input signal
        fs : float
            Sampling frequency of the signal in Hz
        method : str, optional
            'fft' or 'welch', by default 'fft'
        nperseg : int, optional
            Length of each segment for Welch's method, by default None

        Returns
        -------
        float
           Dominant frequency of the signal
        
        Raises
        ------
        ValueError
            If a proper method is not selected
        """
        
        # Compute power spectral density
        if method == 'fft':
            freqs, Pxx = signal.periodogram(x, fs)
        elif method == 'welch':
            freqs, Pxx = signal.welch(x, fs, nperseg=nperseg)
        else:
            raise ValueError('Invalid method: %s' % method)
        
        # Find index of maximum PSD value
        max_psd_idx = np.argmax(Pxx)
        dominant_freq = freqs[max_psd_idx]
        
        return dominant_freq
    
    def dominant_frequency_ecg(
        self, 
        ecg: np.ndarray,
        fs: float,
        method: str = 'fft', 
        nperseg: int = None, 
    ) -> np.ndarray:
        
        m, _, lead = ecg.shape
        ecg_dominant_frequency = np.zeros((m, lead))
        for i in range(m):
            for j in range(lead):
                ecg_dominant_frequency[i, j] = self.dominant_frequency(
                    x=ecg[i, :, j].ravel(),
                    fs=fs, 
                    method=method, 
                    nperseg=nperseg
                )
        
        return ecg_dominant_frequency
    
    def spectral_entropy(
        self,
        x: np.ndarray, 
        fs: float, 
        method: str = 'fft', 
        nperseg: int = None, 
        normalize: bool = True
    ) -> float:
        """Compute spectral entropy of a signal
        
        The method takes as input a signal x, its sampling frequency fs, 
        and several optional parameters, including the method for computing 
        the power spectral density (method), the segment length for Welch's 
        method (nperseg), and whether to normalize the spectral density (normalize). 
        By default, the function uses the FFT method to compute 
        the power spectral density.

        The function first computes the power spectral density using the specified
        method and normalizes it if normalize=True. It then computes the spectral 
        entropy using the formula -sum(Pxx * log2(Pxx)), where Pxx is the normalized 
        power spectral density.

        Parameters
        ----------
        x : np.ndarray
            Input signal
        fs : float
           Sampling frequency of the signal in Hz
        method : str, optional
            'fft' or 'welch', by default 'fft'
        nperseg : int, optional
            Length of each segment for Welch's method, by default None
        normalize : bool, optional
            Normalize the spectral density, by default True

        Returns
        -------
        float
            Spectral entropy of the signal

        Raises
        ------
        ValueError
            If a proper method is not selected
        """

        # Compute power spectral density
        if method == 'fft':
            _, Pxx = signal.periodogram(x, fs)
        elif method == 'welch':
            _, Pxx = signal.welch(x, fs, nperseg=nperseg)
        else:
            raise ValueError('Invalid method: %s' % method)

        # Normalize the spectral density
        if normalize:
            # Pxx /= np.sum(Pxx)
            out = np.ones(Pxx.shape)
            sumpxx = np.sum(Pxx)
            Pxx = np.divide(Pxx, sumpxx, out=out, where=(sumpxx != 0))

        # Compute spectral entropy
        out = np.ones(Pxx.shape)
        log2pxx = np.log2(Pxx, out=out, where=(Pxx > 0))
        spectral_entropy = -np.sum(Pxx * log2pxx)

        return spectral_entropy
    
    def spectral_entropy_ecg(
        self, 
        ecg: np.ndarray,
        fs: float,
        method: str = 'fft', 
        nperseg: int = None, 
        normalize: bool = True
    ) -> np.ndarray:
        
        m, _, lead = ecg.shape
        ecg_spectral_entropy = np.zeros((m, lead))
        for i in range(m):
            for j in range(lead):
                ecg_spectral_entropy[i, j] = self.spectral_entropy(
                    x=ecg[i, :, j].ravel(),
                    fs=fs, 
                    method=method, 
                    nperseg=nperseg, 
                    normalize=normalize
                )
        
        return ecg_spectral_entropy