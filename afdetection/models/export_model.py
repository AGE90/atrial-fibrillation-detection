import joblib
import afdetection.utils.paths as path
from sklearn.base import BaseEstimator

class ExportModel:
    
    def model_export(self, model: BaseEstimator, score: float) -> None:
        """_summary_
        Parameters
        ----------
        model : BaseEstimator
            _description_
        score : float
            _description_
        """
        
        print('Model score: {}'.format(score))
        models_DIR = path.models_dir('best_model.pkl')
        joblib.dump(model, models_DIR)