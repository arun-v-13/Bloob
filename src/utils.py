import os
import sys

from src.exception import CustomException

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
        