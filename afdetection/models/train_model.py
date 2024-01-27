import numpy as np
import pandas as pd
import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous

from afdetection.models.export_model import ExportModel

class TrainModel:
    
    def __init__(self) -> None:

        self.estimators = {
            'RNDFOREST' : RandomForestClassifier(),
            'XGBOOST' : XGBClassifier()
        }
        
        self.param_grids = {
            'RNDFOREST' : {
                'estimator__n_estimators' : Integer(10, 400),
                'estimator__max_depth' : Categorical([None, 5, 10, 20]),
                'estimator__min_samples_split' : Integer(2, 10),
                'estimator__min_samples_leaf' : Integer(1, 4),
                'estimator__max_features' : Categorical(['sqrt', 'log2'])
            },
            'XGBOOST' : {
                'estimator__n_estimators' : Integer(40, 400),
                'estimator__learning_rate' : Continuous(0.01, 0.2),
                'estimator__max_depth' : Integer(3, 10),
                'estimator__subsample' : Continuous(0.5, 1),
                'estimator__colsample_bytree' : Continuous(0.5, 1),
                'estimator__gamma': Integer(0, 1),
                'estimator__reg_lambda' : Continuous(0, 10),
                'estimator__reg_alpha' : Continuous(0, 10),
                'estimator__min_child_weight' : Continuous(0, 10)
            }
        }
        
        
    def genopt_training(self, X: pd.DataFrame, y: pd.Series) -> None:
        
        assert not X.empty, "Input features X cannot be empty"
        assert not y.empty, "Output values y cannot be empty"
        assert X.shape[0] == y.shape[0], "Input features and output values must have the same number of samples"
        assert not X.isnull().values.any(), "Input features X contains missing values"
        assert not y.isnull().values.any(), "Output values y contains missing values"
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        best_score = 0
        best_model = None
               
        for name, estimator in self.estimators.items():
            logging.info(f"Performing evolutionary optimization over hyperparameters for {name}...")
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95, svd_solver='full')),
                ('estimator', estimator)
            ])
            
            evolved_estimator = GASearchCV(
                estimator=pipeline,
                cv=3,
                scoring='accuracy',
                param_grid=self.param_grids[name],
                n_jobs=-1,
                verbose=True,
                population_size=15,
                generations=8
            ).fit(X, y)
            
            score = evolved_estimator.best_score_
            
            logging.info(f"Best {name} model has accuracy score of {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_model = evolved_estimator.best_estimator_
        
        logging.info(f"Exporting best model with accuracy score of {best_score:.2f}")  
        export = ExportModel()
        export.model_export(best_model, best_score)