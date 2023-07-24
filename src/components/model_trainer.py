from src.entity import artifact_entity, config_entity
from src.exception import CustomException
from src.logger import logging
from scipy.stats import ks_2samp
from typing import Optional
import os, sys
from src import utils
'''from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier'''
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score


class ModelTrainer:
    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact):
            try:
                self.model_trainer_config = model_trainer_config
                self.data_transformation_artifact = data_transformation_artifact
            except Exception as e:
                raise CustomException(e,sys)
            
    def train_model(self,x,y):
        try:
            svc = SVC()
            return (svc.fit(x,y))
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_model_trainer(self,):  # Output of data transformation
        try:
            logging.info("Loading train and test data") 
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info("Splitting data into train and test")
            X_train,y_train, X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
            logging.info(f"Shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")


            logging.info("Model train started")
            model = self.train_model(X_train,y_train)
            # After model training
            logging.info("Model train done")
            logging.info(f"Trained model: {model}")

            logging.info("Predicting on train set")
            y_pred_train = model.predict(X_train)
            
            train_accuracy = accuracy_score(y_train, y_pred_train)
            train_precision = precision_score(y_train, y_pred_train)
            train_recall = recall_score(y_train, y_pred_train)
            train_f1_score = f1_score(y_train, y_pred_train)

            logging.info(f"Train score: accuracy={train_accuracy} precision={train_precision} recall={train_recall} f1_score={train_f1_score}")

            logging.info("Predicting on test set")
            y_pred_test = model.predict(X_test)
           
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test)
            test_recall = recall_score(y_test, y_pred_test)
            test_f1_score = f1_score(y_test, y_pred_test)

            logging.info(f"Test score: accuracy={test_accuracy} precision={test_precision} recall={test_recall} f1_score={test_f1_score}")

            logging.info("Saving model")
            utils.save_object(file_path=self.model_trainer_config.model_path,obj=model)

            logging.info('Artifact started ')
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path
            )
            logging.info('Returning model artifact')
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)