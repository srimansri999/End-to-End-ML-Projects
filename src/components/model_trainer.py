import os
import sys
from dataclasses import dataclass
from pathlib import Path

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(str(PROJECT_ROOT), 'artifacts', 'model.pkl')

class ModelTrainer:
    def __init__ (self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splittingh train and test data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models ={
                'Random Forest':RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'Linear Regression':LinearRegression(),
                'K-Neighbors Regressor':KNeighborsRegressor(),
                'XGBoost Regressor':XGBRegressor(),
                'CatBoosting Regressor':CatBoostRegressor(),
                'AdaBoost Regressor':AdaBoostRegressor()
            }

            params = {
                'Random Forest':{
                    'n_estimators':[10,20,50,100,150],
                    'criterion':['squared_error','absolute_error','poisson'],
                    'max_depth':[None, 10,20,30],
                    'min_samples_leaf':[1,2,4],
                    'bootstrap':[True,False]
                },
               

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20, 30, 50],
                    "min_samples_split": [2, 5, 10, 20],
                    "min_samples_leaf": [1, 2, 4, 8],
                    "max_features": [None, "sqrt", "log2"]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 4, 5, 6, 8],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.6, 0.8, 1.0],
                    "max_features": ["sqrt", "log2", None]
                },

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9, 11, 15],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "leaf_size": [20, 30, 40, 50],
                    "p": [1, 2]
                },

                "XGBoost Regressor": {
                    "n_estimators": [100, 200, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7, 10],
                    "min_child_weight": [1, 3, 5, 7],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "gamma": [0, 0.1, 0.2, 0.3],
                    "reg_alpha": [0, 0.01, 0.1, 1],
                    "reg_lambda": [1, 2, 5, 10]
                },

                "CatBoosting Regressor": {
                    "iterations": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "depth": [4, 6, 8, 10],
                    "l2_leaf_reg": [1, 3, 5, 7, 9],
                    "loss_function": ["RMSE", "MAE"],
                    "bagging_temperature": [0, 1, 3, 5]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200, 500],
                    "learning_rate": [0.001, 0.01, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"]
                }
}

            

            model_report, best_models = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models,
                params
            )

            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[ 
                list(model_report.values()).index(best_model_score)
            ]

            best_model = best_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No Best Model Found')
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted= best_model.predict(X_test)
            r2score = r2_score(y_test, predicted)
            return r2score

        except Exception as e:
            raise CustomException(e,sys)
            
