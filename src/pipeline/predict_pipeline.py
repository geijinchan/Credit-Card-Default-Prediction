import pandas as pd
import numpy as np
import os, sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class PredictPipeline:
    def __init__(self, model_path, data_transformer_path, encoder_path):
        try:
            self.path_to_model = model_path
            self.path_to_preprocessor = data_transformer_path
            self.encoder = encoder_path
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(self, features):
        try:
            model_path = self.path_to_model
            data_preprocessor_path = self.path_to_preprocessor
        
            model = load_object(file_path=model_path) 
            data_transformer = load_object(file_path=data_preprocessor_path)
          
            features.drop(columns=['ID'], inplace=True)
            logging.info("Loading model and transformer file done")

            data_scaled = data_transformer.transform(features)
            logging.info("Scaled data")
            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            raise CustomException(e,sys)
       

class CustomData:
    def __init__(self, ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
                 PAY_SEPT, PAY_AUG, PAY_JULY, PAY_JUNE, PAY_MAY, PAY_APRIL,
                 BILL_AMT_SEPT, BILL_AMT_AUG, BILL_AMT_JULY, BILL_AMT_JUNE,
                 BILL_AMT_MAY, BILL_AMT_APRIL,
                 PAY_AMT_SEPT, PAY_AMT_AUG, PAY_AMT_JULY, PAY_AMT_JUNE,
                 PAY_AMT_MAY, PAY_AMT_APRIL):
        self.ID = ID
        self.LIMIT_BAL = LIMIT_BAL
        self.SEX = SEX
        self.EDUCATION = EDUCATION
        self.MARRIAGE = MARRIAGE
        self.AGE = AGE
        self.PAY_SEPT = PAY_SEPT
        self.PAY_AUG = PAY_AUG
        self.PAY_JULY = PAY_JULY
        self.PAY_JUNE = PAY_JUNE
        self.PAY_MAY = PAY_MAY
        self.PAY_APRIL = PAY_APRIL
        self.BILL_AMT_SEPT = BILL_AMT_SEPT
        self.BILL_AMT_AUG = BILL_AMT_AUG
        self.BILL_AMT_JULY = BILL_AMT_JULY
        self.BILL_AMT_JUNE = BILL_AMT_JUNE
        self.BILL_AMT_MAY = BILL_AMT_MAY
        self.BILL_AMT_APRIL = BILL_AMT_APRIL
        self.PAY_AMT_SEPT = PAY_AMT_SEPT
        self.PAY_AMT_AUG = PAY_AMT_AUG
        self.PAY_AMT_JULY = PAY_AMT_JULY
        self.PAY_AMT_JUNE = PAY_AMT_JUNE
        self.PAY_AMT_MAY = PAY_AMT_MAY
        self.PAY_AMT_APRIL = PAY_AMT_APRIL

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dic = {
                "ID": [self.ID],
                "LIMIT_BAL": [self.LIMIT_BAL],
                "SEX": [self.SEX],
                "EDUCATION": [self.EDUCATION],
                "MARRIAGE": [self.MARRIAGE],
                "AGE": [self.AGE],
                "PAY_SEPT": [self.PAY_SEPT],
                "PAY_AUG": [self.PAY_AUG],
                "PAY_JULY": [self.PAY_JULY],
                "PAY_JUNE": [self.PAY_JUNE],
                "PAY_MAY": [self.PAY_MAY],
                "PAY_APRIL": [self.PAY_APRIL],
                "BILL_AMT_SEPT": [self.BILL_AMT_SEPT],
                "BILL_AMT_AUG": [self.BILL_AMT_AUG],
                "BILL_AMT_JULY": [self.BILL_AMT_JULY],
                "BILL_AMT_JUNE": [self.BILL_AMT_JUNE],
                "BILL_AMT_MAY": [self.BILL_AMT_MAY],
                "BILL_AMT_APRIL": [self.BILL_AMT_APRIL],
                "PAY_AMT_SEPT": [self.PAY_AMT_SEPT],
                "PAY_AMT_AUG": [self.PAY_AMT_AUG],
                "PAY_AMT_JULY": [self.PAY_AMT_JULY],
                "PAY_AMT_JUNE": [self.PAY_AMT_JUNE],
                "PAY_AMT_MAY": [self.PAY_AMT_MAY],
                "PAY_AMT_APRIL": [self.PAY_AMT_APRIL],
            }
            return pd.DataFrame(custom_data_input_dic)
        except Exception as e:
            raise CustomException(e, sys)