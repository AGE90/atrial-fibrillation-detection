import numpy as np
from scipy import signal

class SignalPreprocessing:
    
    def mean_removal(self, ecg: np.ndarray) -> np.ndarray:
        """Mean removal

        Parameters
        ----------
        ecg : np.ndarray
            Electrocardiograms array

        Returns
        -------
        np.ndarray
            Transformed electrocardiograms array
        """
        
        ecg = np.transpose(ecg, (1, 0, 2)) - ecg.mean(axis=1)
        ecg = np.transpose(ecg, (1, 0, 2))
        
        return ecg
    
    def wander_removal(self, ecg: np.ndarray, fs: float, fc: float = 0.5, order: int = 4) -> np.ndarray:
        """Wander removal

        Parameters
        ----------
        ecg : np.ndarray
           Electrocardiograms array
        fs : float
            Sampling frequency (Hz)
        fc : float, optional
            Cutoff frequency, by default 0.5 Hz
        order : int, optional
            High-pass butterworth filter order, by default 4

        Returns
        -------
        np.ndarray
             Filtered electrocardiograms array
        """
        
        # Define the high-pass filter
        fc = 0.5  # Cutoff frequency
        order = 4  # Filter order
        b, a = signal.butter(order, fc/(fs/2), 'highpass')

        # Apply the filter to remove baseline wander
        filtered_ecg = signal.filtfilt(b, a, ecg, axis=1)
        
        return filtered_ecg
    
    def moving_window_integration(self, x: np.ndarray, w: int) -> np.ndarray:
        """Moving window integration

        Parameters
        ----------
        x : np.ndarray
            Single electrocardiogram array
        w : int
            Sliding window length

        Returns
        -------
        np.ndarray
            Transformed electrocardiogram array
        """
        
        return np.convolve(x, np.ones(w), 'same') / w
    
    def band_pass_filtering(
        self,
        ecg: np.ndarray, 
        fs: float, 
        fc_low: float = 5, 
        fc_high: float = 12, 
        order: int = 4
        ) -> np.ndarray:
        """Band-pass

        Parameters
        ----------
        ecg : np.ndarray
            Electrocardiograms array
        fs : float
            Sampling frequency (Hz)
        fc_low : float, optional
            Cutoff frequency for the low pass, by default 5 Hz
        fc_high : float, optional
            Cutoff frequency for the high pass, by default 12 Hz
        order : int, optional
            Band-pass butterworth filter order, by default 4, by default 4

        Returns
        -------
        np.ndarray
            Filtered electrocardiograms array
        """
                
        b, a = signal.butter(order, [fc_low/(fs/2), fc_high/(fs/2)], 'bandpass')
        ecg = signal.filtfilt(b, a, ecg, axis=1)
        
        return ecg
    
    def derivative_filtering(self, ecg: np.ndarray) -> np.ndarray:
        """Compute the first derivative

        Parameters
        ----------
        ecg : np.ndarray
            Electrocardiograms array

        Returns
        -------
        np.ndarray
            Transformed electrocardiogram array
        """
        
        return np.gradient(ecg, axis=1)
    
    def squaring(self, ecg: np.ndarray) -> np.ndarray:
        """Square the signal

        Parameters
        ----------
        ecg : np.ndarray
            Electrocardiograms array

        Returns
        -------
        np.ndarray
            Transformed electrocardiogram array
        """
        
        return np.square(ecg)
    
    def smooth_signals(self, ecg: np.ndarray,  w: float) -> np.ndarray:
        """Apply moving-average to smooth the signal

        Parameters
        ----------
        ecg : np.ndarray
            Electrocardiograms array
        w : float
            Moving average window size

        Returns
        -------
        np.ndarray
            Transformed electrocardiogram array
        """
        
        m, n, lead = ecg.shape
        ecg_smooth = np.zeros((m, n, lead))
        for i in range(m):
            for j in range(lead):
                ecg_smooth[i, :, j] = self.moving_window_integration(ecg[i, :, j], w)
        
        return ecg_smooth
    
    def pan_tompkins(self, ecg: np.ndarray, fs: float, w: float) -> np.ndarray:
        """Pan–Tompkins algorithm

        Parameters
        ----------
        ecg : np.ndarray
           Electrocardiograms array
        fs : float
            Sampling frequency (Hz)
        w : float
            Moving average window size

        Returns
        -------
        np.ndarray
            Transformed electrocardiogram array
        """
        
        ecg = self.band_pass_filtering(ecg=ecg, fs=fs)
        ecg = self.derivative_filtering(ecg=ecg)
        ecg = self.squaring(ecg=ecg)
        ecg = self.smooth_signals(ecg=ecg, w=w)
        
        return ecg
        
    
    def normalize(self, ecg: np.ndarray) -> np.ndarray:
        """Normalize signals between [-1, 1]

        Parameters
        ----------
        ecg : np.ndarray
            Electrocardiograms array

        Returns
        -------
        np.ndarray
            Transformed electrocardiogram array
        """
        
        return (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))