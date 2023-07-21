from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    report_file_path: str

@dataclass
class DataTransformationArtifact:
    transform_object_path:str
    transformed_train_path:str
    transformed_test_path:str

@dataclass
class ModelTrainerArtifact:
    model_path: str
    

@dataclass
class ModelEvaluationArtifact:
    # Define properties for model evaluation artifact
    pass

@dataclass
class ModelPusherArtifact:
    # Define properties for model pusher artifact
    pass
