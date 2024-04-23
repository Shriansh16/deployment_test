import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, 'D:\Laptop_Price_Prediction\src')
from logger import *
from exception import *
from utils import *
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformation(self):
        try:
            num_columns=['Ram', 'Weight', 'Touchscreen', 'IPS', 'ppi', 'HDD', 'SSD']
            cat_columns=['Company', 'TypeName', 'CPU_BRAND', 'GPU_BRAND', 'OS']
            preprocessor=ColumnTransformer([('cat_transformer',OneHotEncoder(),cat_columns),('num_transformer',
                                                                                             StandardScaler(),num_columns)])
            logging.info('preprocessing of categorical data and numerical data is done')
            return preprocessor
        except Exception as e:
            logging.info('ERROR OCCURED DURING DATA TRANSFORMATION')
            raise CustomException(e,sys)
    def initiate_data_tranforamation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            preprocessor_obj=self.get_data_transformation()
            input_train_features=train_df.drop(columns=['Price'],axis=1)
            output_train_features=train_df['Price']
            input_test_features=test_df.drop(columns=['Price'],axis=1)
            output_test_features=test_df['Price']
            logging.info('TRANSFORMATION STARTED')
            input_train_trans=preprocessor_obj.fit_transform(input_train_features)
            input_test_trans=preprocessor_obj.transform(input_test_features)
            logging.info('TRANSFORAMTION DONE')          
            logging.info(output_train_features.shape)
            logging.info(input_train_trans.shape)
            input_train_trans = input_train_trans.toarray()
            input_test_trans=input_test_trans.toarray()
            train_arr=np.c_[input_train_trans,np.array(output_train_features)]
            logging.info('1')
            test_arr=np.c_[input_test_trans,np.array(output_test_features)]
            save_object(self.data_transformation_config.preprocessor_path,preprocessor_obj)
            return (train_arr,test_arr)
        except Exception as e:
            logging.info('ERROR OCCURED DURING INITIATION OF DATA TRANSFORAMTION')
            raise CustomException(e,sys)
            


