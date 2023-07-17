import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.entity import config_entity
from src.entity import artifact_entity
from src.utils import get_collection_as_dataframe
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

@dataclass
class DataIngestion:
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data ingestion {'<<'*20 }")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        logging.info('Entering the data ingestion method')
        try:
            logging.info("Exporting collection data as pandas dataframe")
            df = get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name
            )

            logging.info('Save data in feature store')

            # Replace na with nan
            df.replace(to_replace="na", value=np.NAN, inplace=True)

            # Save data in feature store
            logging.info("Create feature store folder if not available")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            logging.info("Save df to feature store folder")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path, index=False, header=True)

            logging.info("Splitting dataset into train set and test set")
            # Split dataset into train set and test set
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size, random_state=42)

            logging.info("Create dataset directory folder if not available")
            # Create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info("Save train and test data to dataset directory")
            # Save train and test data to dataset directory
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            # Prepare artifact
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
