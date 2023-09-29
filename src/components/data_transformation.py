import sys
import os
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path = os.path.join('artifacts' , 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            
            
            # mentioning the features
            numerical_feature_names = ['reading_score', 'writing_score']
            categorical_feature_names = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            logging.info(f'Numerical Features : {numerical_feature_names}')
            logging.info(f'Categorical Features : {categorical_feature_names}'
                         )
            logging.info('Creating preprocessor object for imputing, encoding and scaling of features')
            # defining pipelines for each
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer" , SimpleImputer(strategy="median")),
                    ("scaler" , StandardScaler())
                    ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer" , SimpleImputer(strategy= "most_frequent")),
                    ("encoder" , OneHotEncoder()),
                    ("scaler" , StandardScaler(with_mean= False)) # coz x train is a sparsed matrix with a lot of zeroes due to one hot encoding, hence centering the data becomes hard to do with mean.
                ]
            )

            # executing those pipelines using a column transformer
            preprocessor = ColumnTransformer([("numerical_transformation" , numerical_pipeline, numerical_feature_names),
                ('categorical_transformation' , categorical_pipeline , categorical_feature_names)]
            )

            
            return preprocessor


        except Exception as e:
            raise CustomException(e , sys)
            

    def initiate_data_transformation(self , train_data_path , test_data_path):
        try:
            logging.info('DATA TRANSFORMATION BEGINS...')

            logging.info('Reading train and test data')
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Fetching preprocessor object')
            preprocessing_object = self.get_data_transformer_object()

            logging.info('Dividing data as input and output features')
            target_feature = "math_score"


            x_train_raw = train_df.drop(columns = [target_feature])
            y_train = train_df[target_feature]

            x_test_raw = test_df.drop(columns = [target_feature] )
            y_test = test_df[target_feature]

            logging.info('Applying preprocessor object on input data')
            x_train = preprocessing_object.fit_transform(x_train_raw) # this would be returned in the form of an array
            x_test = preprocessing_object.transform(x_test_raw)
            
            logging.info('Combining input and output data back together again')
            train_arr = np.c_[ x_train , np.array(y_train)] # concatenates two arrays column wise
            test_arr = np.c_ [ x_test , np.array(y_test)]

            logging.info('Saving preprocessor object as a pickle file in artifacts')
            save_object(file_path= self.data_transformation_config.preprocessor_object_file_path,
                        object= preprocessing_object)

            logging.info('DATA TRANSFORMATION COMPLETED.')

            return (train_arr ,
                    test_arr ,
                    self.data_transformation_config.preprocessor_object_file_path)


        except Exception as e:
            raise CustomException(e , sys)
            