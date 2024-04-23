import os
import sys
import pandas as pd
sys.path.insert(0, 'D:\Laptop_Price_Prediction\src')
from logger import logging
from exception import CustomException
from utils import *
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer

if __name__=='__main__':
    obj1=DataIngestion()
    train_path,test_path=obj1.initiate_data_ingestion()
    obj2=DataTransformation()
    train_arr,test_arr=obj2.initiate_data_tranforamation(train_path,test_path)
    obj3=ModelTrainer()
    R2_SCORE=obj3.initiate_model_training(train_arr,test_arr)
    print(R2_SCORE)
