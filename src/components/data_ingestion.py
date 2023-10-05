import sys 
import os


from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation  # for testing purposes
from src.components.model_trainer import ModelTrainer
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:  # the dataclass decorator replaces the init statment in cases where the class is used just to define variables.
    raw_data_path : str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path :str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self): # to get data into fixed path from where components can access.
        logging.info('DATA INGESTION BEGINS...')
        try:
            logging.info('Reading data as DataFrame from external source')
            df = pd.read_csv('notebook\data\stud.csv') # external source of data
            

            logging.info('Creating artifacts folder where data can be stored')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path) , exist_ok= True)
            
            logging.info(r'Storing raw data in artifacts' )
            df.to_csv(self.ingestion_config.raw_data_path , index = False , header = True)

            logging.info('Initiating train-test split')
            train_set , test_set = train_test_split(df , test_size = 0.2 , random_state = 13)

            logging.info(r'Storing train and test data in artifacts')
            train_set.to_csv(self.ingestion_config.train_data_path , index = False , header = True)
            test_set.to_csv(self.ingestion_config.test_data_path , index = False , header = True)

            logging.info('DATA INGESTION COMPLETED')

            return self.ingestion_config.train_data_path , self.ingestion_config.test_data_path
        
        except Exception as e:
            raise CustomException(e, sys)
            
            

if __name__ == '__main__':
    object = DataIngestion()
    train_data_path , test_data_path = object.initiate_data_ingestion()

    preprocessor = DataTransformation()
    train_array , test_array , preprocessor_path = preprocessor.initiate_data_transformation(train_data_path= train_data_path,
                                              test_data_path= test_data_path)
    
    trainer = ModelTrainer()
    print(trainer.initiate_model_trainer(train_array= train_array,
                                   test_array= test_array))
    



