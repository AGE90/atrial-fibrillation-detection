import numpy as np
import pandas as pd

class MakeDataset:
    
    def read_from_csv(self, path: str) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        path : str
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        
        return pd.read_csv(path, sep=';', header=0, index_col=0)
    
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