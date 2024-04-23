import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, 'D:\Laptop_Price_Prediction\src')
from logger import *
from exception import *
from utils import *
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

@dataclass
class ModelTrainerConfig:
    model_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('initiating training')
            X_train,y_train,X_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            models={'LinearRegressor ':LinearRegression(),'RandomForestRegressor ':RandomForestRegressor(),
                    'AdaBoostRegressor ':AdaBoostRegressor(),'GradientBoostingRegressor ':GradientBoostingRegressor(),
                    'DecisionTreeRegressor ':DecisionTreeRegressor(),'KneighborsRegressor ':KNeighborsRegressor()}
            logging.info('TRAINING STARTED')
            logging.info('MODEL REPORT : ')
            model_report=evaluate_models(X_train,y_train,X_test,y_test,models)
            logging.info(model_report)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            save_object(self.model_trainer_config.model_path,best_model)
            predicted_data=best_model.predict(X_test)
            R2_SCORE=r2_score(y_test,predicted_data)
            return R2_SCORE
        except Exception as e:
            logging.info('ERROR OCCURRED DURING MODEL TRAINING')
            raise CustomException(e,sys)
            
            
            

