import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous

from afdetection.models.export_model import ExportModel

class TrainModel:
    
    def __init__(self) -> None:
        self.estimators = {
            'RNDFOREST' : RandomForestClassifier(),
            'XGBOOST' : XGBClassifier()
        }
        
        # self.param_grids = {
        #     'RNDFOREST' : {
        #         'n_estimators' : np.arange(40, 420, 20),
        #         'max_features' : ['sqrt', 'log2'],
        #         'max_depth' : [10, 20, 30, 40, 50, 60, 70, None],
        #         'min_samples_split' : [2, 5, 10],
        #         'min_samples_leaf' : [1, 2, 4],
        #         'bootstrap' : [True, False]
        #     },
        #     'XGBOOST' : {
        #         'n_estimators' : np.arange(40, 420, 20),
        #         'max_depth' : np.arange(3, 20, 2),
        #         'gamma': np.arange(1, 10),
        #         'alpha' : np.arange(0, 20, 2),
        #         'lambda' : np.arange(0, 1.1, 0.1),
        #         'colsample_bytree' : np.arange(0.5, 1.1, 0.1),
        #         'min_child_weight' : np.arange(0, 10)
        #     }
        # }
        
        self.param_grids = {
            'RNDFOREST' : {
                'n_estimators' : Integer(40, 400),
                'max_features' : Categorical(['sqrt', 'log2']),
                'max_depth' : Integer(10, 70),
                'min_samples_split' : Integer(2, 10),
                'min_samples_leaf' : Integer(1, 4),
                'bootstrap' : Categorical([True, False])
            },
            'XGBOOST' : {
                'n_estimators' : Integer(40, 420),
                'max_depth' : Integer(3, 20),
                'gamma': Integer(1, 10),
                'alpha' : Continuous(0, 20),
                'lambda' : Continuous(0, 1),
                'colsample_bytree' : Continuous(0.5, 1),
                'min_child_weight' : Integer(0, 10)
            }
        }
        
    def grid_training(self, X: pd.DataFrame, y: pd.Series) -> None:
        
              
        best_score = 999
        best_model = None
        
        for name, estimator in self.estimators.items():
            grid = GridSearchCV(
                estimator=estimator,
                param_grid=self.param_grids[name],
                cv=3
            ).fit(X, y)
            
            score = np.abs(grid.best_score_)
            
            if score < best_score:
                best_score = score
                best_model = grid.best_estimator_
        
        export = ExportModel()
        export.model_export(best_model, best_score)
        
    def genopt_training(self, X: pd.DataFrame, y: pd.Series) -> None:
        
        best_score = 999
        best_model = None
        
        for name, estimator in self.estimators.items():
            evolved_estimator = GASearchCV(
                estimator=estimator,
                cv=3,
                scoring='accuracy',
                param_grid=self.param_grids[name],
                n_jobs=-1,
                verbose=True,
                population_size=10,
                generations=10
            ).fit(X, y)
            
            score = np.abs(evolved_estimator.best_score_)
            
            if score < best_score:
                best_score = score
                best_model = evolved_estimator.best_estimator_
                
        export = ExportModel()
        export.model_export(best_model, best_score)