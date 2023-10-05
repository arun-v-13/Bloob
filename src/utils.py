import os
import sys

from src.exception import CustomException

from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import dill # library to create and manage pickle files

def save_object(file_path , object):
    try:
        directory_name = os.path.dirname(file_path)
        os.makedirs(directory_name , exist_ok= True)

        with open(file_path , "wb") as f:
            dill.dump(object , f)
            
    except Exception as e:
        raise CustomException(e , sys)
    
def evaluate_models(x_train , y_train , x_test , y_test , models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]

            # Training the model
            model.fit(x_train , y_train)

            # Predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # R2 scores
            train_model_score = r2_score(y_train , y_train_pred)
            test_model_score = r2_score(y_test , y_test_pred)

            # Appending only test model score to report
            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
    


        