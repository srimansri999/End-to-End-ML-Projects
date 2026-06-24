import os
import sys

import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        best_estimators = {}

        for model_name, model in models.items():
            para = param[model_name]

            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=para,
                n_iter=20,
                cv=3,
                n_jobs=-1,
                random_state=42
            )

            rs.fit(X_train, y_train)

            best_model = rs.best_estimator_
            best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            best_estimators[model_name] = best_model

            print(
                f"{model_name} | Train R2: {train_model_score:.4f} | "
                f"Test R2: {test_model_score:.4f}"
            )

        return report, best_estimators

    except Exception as e:
        raise CustomException(e, sys)