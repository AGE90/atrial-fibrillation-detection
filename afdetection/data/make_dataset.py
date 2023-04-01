import numpy as np
import pandas as pd

class MakeDataset:
    
    def read_from_csv(self, path: str, sep: str = ',') -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        path : str
            _description_
        sep : str, optional
            _description_, by default ','

        Returns
        -------
        pd.DataFrame
            _description_
        """
        
        return pd.read_csv(path, sep=sep, header=0)
    
    def read_from_np(self, path: str) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        path : str
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        
        return np.load(path)