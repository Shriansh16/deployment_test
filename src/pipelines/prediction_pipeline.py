import os
import sys
import pandas as pd
sys.path.insert(0, 'D:\Laptop_Price_Prediction\src')
from logger import logging
from exception import CustomException
from utils import *
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation

class Predict_Pipeline:
    def __init__(self):
        pass
    def predict(self,feature):
        try:

            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            scaled_data=preprocessor.transform(feature)
            pred=model.predict(scaled_data)
            return pred
        except Exception as e:
            logging.info('ERROR OCCURED DURING PREDICTION')
            raise CustomException(e,sys)

class CustomData:
        def __init__(self,
                 Company:str,
                 TypeName:str,
                 Ram:int,
                 Weight:float,
                 Touchscreen:int,
                 IPS:int,
                 ppi:float,
                 CPU_BRAND:str,
                 HDD:int,
                 SSD:int,
                 GPU_BRAND:str,
                 OS:str):
        
            self.Company=Company
            self.TypeName=TypeName
            self.Ram=Ram
            self.Weight=Weight
            self.Touchscreen=Touchscreen
            self.IPS=IPS
            self.ppi=ppi
            self.CPU_BRAND = CPU_BRAND
            self.HDD = HDD
            self.SSD = SSD
            self.GPU_BRAND=GPU_BRAND
            self.OS=OS

        def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                'Company':[self.Company],
                'TypeName':[self.TypeName],
                'Ram':[self.Ram],
                'Weight':[self.Weight],
                'Touchscreen':[self.Touchscreen],
                'IPS':[self.IPS],
                'ppi':[self.ppi],
                'CPU_BRAND':[self.CPU_BRAND],
                'HDD':[self.HDD],
                'SSD':[self.SSD],
                'GPU_BRAND':[self.GPU_BRAND],
                'OS':[self.OS]
                 }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise CustomException(e,sys)