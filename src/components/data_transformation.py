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
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class DataTransformation:
    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        # Add any other initialization code here

    # Rest of the class code...


    @classmethod
    def get_data_tranformer_object(cls)->Pipeline:
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
            logging.info(f"ID columns dropper {train_df.shape}")  # 24 features
            
            logging.info("Obtaining preprocessing object")
            
            target_column = "defaults"

            # Selecting feature for train dataframe
            input_feature_train_df = train_df.drop(target_column,axis=1)
            target_feature_train_df = train_df[target_column]
            logging.info(f"Input features training {input_feature_train_df.shape}")
            logging.info(f"Input features testing{input_feature_train_df.columns}")

            # Selecting target feature for train and test dataframe
            input_feature_test_df = test_df.drop(target_column,axis=1)
            target_feature_test_df = test_df[target_column]
            logging.info(f"Input features testing{input_feature_test_df.shape}")
            logging.info(f"Input features testing{input_feature_test_df.columns}")

             # Transform on target columns
            encoder = OneHotEncoder(sparse=False)
            target_feature_train_arr = encoder.fit_transform(target_feature_train_df.values.reshape(-1, 1))
            target_feature_test_arr = encoder.transform(target_feature_test_df.values.reshape(-1, 1))

            # Transform on input features
            logging.info(f"Fit transform started for training")
            transformation_pipeline = DataTransformation.get_data_tranformer_object()
            transformation_pipeline.fit(input_feature_train_df)
            
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)
            logging.info(f"after transformation columns are {input_feature_train_arr}")

            smt = SMOTETomek(random_state=42)
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            
            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")


            # Combine input features and encoded target columns
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            utils.save_object(
                file_path=self.data_transformation_config.transform_object_path,
                obj=transformation_pipeline
            )
            
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)
            
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            obj=encoder)
        
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path = self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path
            )
            logging.info("Done and check the transformer.pkl file in the artifact direct")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)
