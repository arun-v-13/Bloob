import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor , GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.logger import logging 
from src.exception import CustomException
from src.utils import save_object , evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array , test_array):
        try:
            logging.info('MODEL TRAINING BEGINS...')

            logging.info('Splitting data into input and output features')
            x_train = train_array[: , :-1] # 2D array slicing [rows , columns]
            y_train = train_array[: , -1]
            x_test = test_array[: , :-1]
            y_test = test_array[: , -1]

            models = {
                "Linear Regression" : LinearRegression(),
                "K Nearest Neighbors" : KNeighborsRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Ada Boost" : AdaBoostRegressor(),
                "Gradient Boost" : GradientBoostingRegressor(),
                "XG Boost" : XGBRegressor(),
                "Cat Boost" : CatBoostRegressor(verbose= False)
            }
             
            logging.info('Training each model and Evaluating its R2 score')
            models_report : dict = evaluate_models(x_train ,
                                                   y_train,
                                                   x_test,
                                                   y_test,
                                                   models)
            
            logging.info('Figuring out the Best Model')
            
            best_model_name , best_model_score = sorted([(k,v) for k,v in models_report.items() ] ,
                                                         key = lambda x : x[1])[-1]
            
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise Exception('No Best Model Found , all scores are below 0.6')
            
            logging.info(f'{best_model_name} has score {best_model_score} which is the highest for all models')
            
            logging.info('Saving model object as a pickle file in artifacts')
            save_object(file_path = self.model_trainer_config.trained_model_file_path,
                        object = best_model )
            
            logging.info('MODEL TRAINING COMPLETED')
            return best_model_name ,best_model_score
    
        except Exception as e:
            raise CustomException(e , sys)




