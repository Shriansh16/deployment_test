import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.insert(0, 'D:\Laptop_Price_Prediction\src')
from logger import *
from exception import *
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join('artifacts','raw.csv')
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            raw_data=pd.read_csv('notebooks\laptop_cleaned_dataset.csv')
            logging.info('dataset is successfully taken')
            train_data,test_data=train_test_split(raw_data,test_size=0.20,random_state=42)
            logging.info('train test split is completed')
            logging.info('saving dataset')
            raw_data.drop('Unnamed: 0',axis=1,inplace=True)
            train_data.drop('Unnamed: 0',axis=1,inplace=True)
            test_data.drop('Unnamed: 0',axis=1,inplace=True)
            raw_data.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info('dataset successfully saved')
            return (self.data_ingestion_config.train_data_path,self.data_ingestion_config.test_data_path)
        except Exception as e:
            logging.info('ERROR OCCURED IN DATA INGESTION')
            raise CustomException(e,sys)





