import sys
from src.exception import CustomException
from src.entity import config_entity, artifact_entity
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.ModelPusher import ModelPusher



if __name__ == "__main__":
    try:
        # Initialize the training pipeline config
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        # Data Ingestion
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # Data Validation
        data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,
                                         data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()

        # Data Transformation
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                 data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation(
                                        train_path=data_ingestion_artifact.train_file_path,
                                        test_path=data_ingestion_artifact.test_file_path
)

        # Model Trainings
        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                     data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        # Model Pusher
        model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config)
        
        model_pusher = ModelPusher(model_pusher_config=model_pusher_config, 
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact)

        model_pusher_artifact = model_pusher.initiate_model_pusher()

    except Exception as e:
        raise CustomException(e, sys)
