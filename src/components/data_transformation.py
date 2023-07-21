import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.entity import artifact_entity,config_entity
from src.exception import CustomException
from src.logger import logging
import os
import sys
import numpy as np
from src import utils
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import TARGET_COLUMN


class DataTransformation:
    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        # Add any other initialization code here

    # Rest of the class code...



    def get_data_tranformer_object(self):
        try:
            '''
            This function is responsible for data transformation
            '''
            num_columns = ['LIMIT_BAL', 'AGE', 'PAY_SEPT',
                            'PAY_AUG', 'PAY_JULY', 'PAY_JUNE', 'PAY_MAY', 'PAY_APRIL',
                            'BILL_AMT_SEPT', 'BILL_AMT_AUG', 'BILL_AMT_JULY', 'BILL_AMT_JUNE',
                            'BILL_AMT_MAY', 'BILL_AMT_APRIL', 'PAY_AMT_SEPT', 'PAY_AMT_AUG',
                            'PAY_AMT_JULY', 'PAY_AMT_JUNE', 'PAY_AMT_MAY', 'PAY_AMT_APRIL']

            
            cat_columns = ['SEX', 'EDUCATION', 'MARRIAGE']

            num_pipeline = Pipeline(
                steps=[
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("one-hot-encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_columns),
                    ("cat_pipeline",cat_pipeline,cat_columns)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading training and testing data done")
            rename_dict = {
                'PAY_0': 'PAY_SEPT',
                'PAY_2': 'PAY_AUG',
                'PAY_3': 'PAY_JULY',
                'PAY_4': 'PAY_JUNE',
                'PAY_5': 'PAY_MAY',
                'PAY_6': 'PAY_APRIL',
                'PAY_AMT1': 'PAY_AMT_SEPT',
                'PAY_AMT2': 'PAY_AMT_AUG',
                'PAY_AMT3': 'PAY_AMT_JULY',
                'PAY_AMT4': 'PAY_AMT_JUNE',
                'PAY_AMT5': 'PAY_AMT_MAY',
                'PAY_AMT6': 'PAY_AMT_APRIL',
                'BILL_AMT1': 'BILL_AMT_SEPT',
                'BILL_AMT2': 'BILL_AMT_AUG',
                'BILL_AMT3': 'BILL_AMT_JULY',
                'BILL_AMT4': 'BILL_AMT_JUNE',
                'BILL_AMT5': 'BILL_AMT_MAY',
                'BILL_AMT6': 'BILL_AMT_APRIL',
                'default.payment.next.month':'defaults'
            }

            train_df.rename(columns=rename_dict, inplace=True)
            test_df.rename(columns=rename_dict, inplace=True)
            logging.info("Rename done")

            train_df.drop(columns=['ID'], inplace=True)
            test_df.drop(columns=['ID'], inplace=True)
            logging.info(f"ID columns dropper ")

            target_column = "defaults"
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_tranformer_object()
            target_column = "defaults"

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column].to_frame()

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column].to_frame()


            logging.info(f"Fit transform started for training")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info("Transform started for testing")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            target_feature_train_arr = np.array(target_feature_train_df).flatten()  # Flatten the target array
            target_feature_test_arr = np.array(target_feature_test_df).flatten()

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_arr)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_arr)]

            utils.save_object(
                file_path=self.data_transformation_config.transform_object_path,
                obj=preprocessing_obj
            )
            
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)
        
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path = self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                
            )
            logging.info("Done and check the transformer.pkl file in the artifact direct")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)
