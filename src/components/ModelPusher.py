import os
import sys
from src.exception import CustomException
from src.entity import config_entity, artifact_entity
from src.utils import load_object, save_object
from src.logger import logging
from src.pipeline.latest_files_function import ModelResolver
from src.entity.config_entity import MODEL_FILE_NAME, TRANSFORMER_OBJECT_FILE_NAME, TARGET_ENCODER_OBJECT_FILE_NAME


class ModelPusher:
    def __init__(self, model_pusher_config: config_entity.ModelPusherConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_pusher(self) -> artifact_entity.ModelPusherArtifact:
        try:
            # Load objects
            logging.info(f"Loading transformer model and target encoder")
            transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path=self.model_trainer_artifact.model_path)
            target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            # Save objects in the model pusher directory
            logging.info(f"Saving model into model pusher directory")
            os.makedirs(self.model_pusher_config.pusher_model_dir, exist_ok=True)
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)

            # Save objects in the saved model dir
            logging.info(f"Saving model in saved model dir")
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            target_encoder_path = self.model_resolver.get_latest_save_target_encoder_path()

            os.makedirs(os.path.dirname(transformer_path), exist_ok=True)
            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)
            save_object(file_path=target_encoder_path, obj=target_encoder)

            model_pusher_artifact = artifact_entity.ModelPusherArtifact(
                latest_transformer_path=transformer_path,
                latest_model_path=model_path,
                latest_encoder_path=target_encoder_path
            )
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e, sys)
