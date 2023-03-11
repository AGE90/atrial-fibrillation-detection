import numpy as np
from scipy import signal

class BuildFeatures:
    
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
           Sampling frequency of the signal
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
            Pxx /= np.sum(Pxx)

        # Compute spectral entropy
        spectral_entropy = -np.sum(Pxx * np.log2(Pxx))

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