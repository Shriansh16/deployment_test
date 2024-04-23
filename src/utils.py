import os
import sys
import pickle
sys.path.insert(0, 'D:\Laptop_Price_Prediction\src')
from logger import logging
from exception import CustomException
from sklearn.metrics import r2_score

def save_object(path,object):
    try:
        dir_path=os.path.dirname(path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(path,'wb') as file_obj:
            pickle.dump(object,file_obj)
    except Exception as e:
        logging.info('ERROR OCCURED IN SAVING THE OBJECT')
        raise CustomException(e,sys)

def load_object(path):
    try:
        with open(path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('ERROR OCCURED DURING LOADING THE OBJECT')
        raise CustomException(e,sys)
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            score=r2_score(y_test,y_pred)
            report[list(models.keys())[i]]=score
        return report
    except Exception as e:
        logging.info('ERROR OCCURED DURING EVALUATING THE MODELS')
        raise CustomException(e,sys)