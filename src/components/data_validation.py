from src.entity import artifact_entity, config_entity
from src.exception import CustomException
from src.logger import logging
from scipy.stats import ks_2samp
from typing import Optional
import os, sys
import pandas as pd
from src import utils
import numpy as np
from src.config import TARGET_COLUMN
from sklearn.ensemble import IsolationForest


class DataValidation:
    def __init__(self, data_validation_config: config_entity.DataValidationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>' * 20} Data Validation {'<<' * 20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = {}  # Dictionary to store validation errors
        except Exception as e:
            raise CustomException(e, sys)

    def drop_missing_values_columns(self, df: pd.DataFrame, report_key_name: str) -> Optional[pd.DataFrame]:
        """
        This function drops columns with missing values above a specified threshold.

        Args:
            df: Pandas DataFrame
            report_key_name: Key name for the validation report

        Returns:
            Optional[pd.DataFrame]: DataFrame after dropping columns, or None if no columns are left
        """
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum() / df.shape[0]
            drop_column_names = null_report[null_report > threshold].index
            self.validation_error[report_key_name] = list(drop_column_names)
            df.drop(list(drop_column_names), axis=1, inplace=True)
            if len(df.columns) == 0:
                return None
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def is_required_columns_exists(self, base_df: pd.DataFrame, current_df: pd.DataFrame,
                                   report_key_name: str) -> bool:
        """
        This function checks if all required columns are present in the current DataFrame.

        Args:
            base_df: Base DataFrame with required columns
            current_df: Current DataFrame to check against
            report_key_name: Key name for the validation report

        Returns:
            bool: True if all required columns are present, False otherwise
        """
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: [{base_column}] is not available.")
                    missing_columns.append(base_column)
            if len(missing_columns) > 0:
                self.validation_error[report_key_name] = missing_columns
                return False
            return True
        except Exception as e:
            raise CustomException(e, sys)

    def data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str):
        """
        This function detects data drift between the base DataFrame and the current DataFrame.

        Args:
            base_df: Base DataFrame for comparison
            current_df: Current DataFrame for comparison
            report_key_name: Key name for the drift report
        """
        try:
            drift_report = {}
            base_columns = base_df.columns
            current_columns = current_df.columns
            for base_column in base_columns:
                base_data, current_data = base_df[base_column], current_df[base_column]
                same_distribution = ks_2samp(base_data, current_data)
                if same_distribution.pvalue > 0.05:
                    drift_report[base_column] = {
                        "pvalues": same_distribution.pvalue,
                        "same_distribution": True
                    }
                else:
                    drift_report[base_column] = {
                        "pvalues": same_distribution.pvalue,
                        "same_distribution": False
                    }
            self.validation_error[report_key_name] = drift_report
        except Exception as e:
            raise CustomException(e, sys)

    def handle_outliers(self, df: pd.DataFrame, report_key_name: str) -> pd.DataFrame:
        """
        This function handles outliers in the DataFrame using the Isolation Forest algorithm.

        Args:
            df: DataFrame to handle outliers in
            report_key_name: Key name for the outlier handling report

        Returns:
            pd.DataFrame: DataFrame after handling outliers
        """
        try:
            num_columns = df.select_dtypes(include=np.number).columns
            clf = IsolationForest(contamination=self.data_validation_config.outlier_threshold, random_state=42)
            clf.fit(df[num_columns])
            outlier_pred = clf.predict(df[num_columns])
            df = df[outlier_pred == 1]
            self.validation_error[report_key_name] = num_columns.tolist()
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na": np.NAN}, inplace=True)
            logging.info(f"Replace na value in base df")
            logging.info(f"Drop null values columns from base df")
            base_df = self.drop_missing_values_columns(df=base_df, report_key_name="missing_values_within_base_dataset")

            logging.info(f"Reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"Reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"Drop null values columns from train df")
            train_df = self.drop_missing_values_columns(df=train_df, report_key_name="missing_values_within_train_dataset")
            logging.info(f"Drop null values columns from test df")
            test_df = self.drop_missing_values_columns(df=test_df, report_key_name="missing_values_within_test_dataset")
            
            # If we ecounter with object type datatype then use this and 
            # check utils.convert_columns_fload
            exclude_columns = [TARGET_COLUMN]
            base_df = utils.convert_columns_float(df=base_df, exclude_columns=exclude_columns)
            train_df = utils.convert_columns_float(df=train_df, exclude_columns=exclude_columns)
            test_df = utils.convert_columns_float(df=test_df, exclude_columns=exclude_columns)

            logging.info(f"Are all required columns present in train df?")
            train_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=train_df,
                                                                      report_key_name="missing_columns_within_train_dataset")
            logging.info(f"Are all required columns present in test df?")
            test_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=test_df,
                                                                     report_key_name="missing_columns_within_test_dataset")

            if train_df_columns_status:
                logging.info(f"All columns are available in the train df, hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data_drift_within_train_dataset")
            if test_df_columns_status:
                logging.info(f"All columns are available in the test df, hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data_drift_within_test_dataset")

            logging.info("Handling outliers in the train and test datasets")
            train_df = self.handle_outliers(df=train_df, report_key_name="outlier_handling_within_train_dataset")
            test_df = self.handle_outliers(df=test_df, report_key_name="outlier_handling_within_test_dataset")

            logging.info("Write report to YAML file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
                                  data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)


