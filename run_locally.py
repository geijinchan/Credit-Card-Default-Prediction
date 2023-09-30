from flask import Flask, render_template, request
import os, sys
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging
from src.entity import config_entity
from src.pipeline.latest_files_function import ModelResolver
from src.utils import load_object

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get all the form fields from the request
            ID = int(request.form.get('ID'))
            LIMIT_BAL = int(request.form.get('LIMIT_BAL'))
            SEX = int(request.form.get('SEX'))
            EDUCATION = int(request.form.get('EDUCATION'))
            MARRIAGE = int(request.form.get('MARRIAGE'))
            AGE = int(request.form.get('AGE'))
            PAY_0 = int(request.form.get('PAY_0'))
            PAY_2 = int(request.form.get('PAY_2'))
            PAY_3 = int(request.form.get('PAY_3'))
            PAY_4 = int(request.form.get('PAY_4'))
            PAY_5 = int(request.form.get('PAY_5'))
            PAY_6 = int(request.form.get('PAY_6'))
            BILL_AMT1 = int(request.form.get('BILL_AMT1'))
            BILL_AMT2 = int(request.form.get('BILL_AMT2'))
            BILL_AMT3 = int(request.form.get('BILL_AMT3'))
            BILL_AMT4 = int(request.form.get('BILL_AMT4'))
            BILL_AMT5 = int(request.form.get('BILL_AMT5'))
            BILL_AMT6 = int(request.form.get('BILL_AMT6'))
            PAY_AMT1 = int(request.form.get('PAY_AMT1'))
            PAY_AMT2 = int(request.form.get('PAY_AMT2'))
            PAY_AMT3 = int(request.form.get('PAY_AMT3'))
            PAY_AMT4 = int(request.form.get('PAY_AMT4'))
            PAY_AMT5 = int(request.form.get('PAY_AMT5'))
            PAY_AMT6 = int(request.form.get('PAY_AMT6'))

            # Create a CustomData instance with the form data
            data = CustomData(
                ID=ID,
                LIMIT_BAL=LIMIT_BAL,
                SEX=SEX,
                EDUCATION=EDUCATION,
                MARRIAGE=MARRIAGE,
                AGE=AGE,
                PAY_SEPT=PAY_0,  # Use the renamed column names here
                PAY_AUG=PAY_2,
                PAY_JULY=PAY_3,
                PAY_JUNE=PAY_4,
                PAY_MAY=PAY_5,
                PAY_APRIL=PAY_6,
                BILL_AMT_SEPT=BILL_AMT1,
                BILL_AMT_AUG=BILL_AMT2,
                BILL_AMT_JULY=BILL_AMT3,
                BILL_AMT_JUNE=BILL_AMT4,
                BILL_AMT_MAY=BILL_AMT5,
                BILL_AMT_APRIL=BILL_AMT6,
                PAY_AMT_SEPT=PAY_AMT1,
                PAY_AMT_AUG=PAY_AMT2,
                PAY_AMT_JULY=PAY_AMT3,
                PAY_AMT_JUNE=PAY_AMT4,
                PAY_AMT_MAY=PAY_AMT5,
                PAY_AMT_APRIL=PAY_AMT6,
            )
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Data fetched from page: {pred_df.columns}")
            print(pred_df)
            
            
            training_pipeline_config = config_entity.TrainingPipelineConfig()
            model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config)
            model_resolver = ModelResolver(model_pusher_config.saved_model_dir)

            # Fetch the latest model, transformer, and encoder paths
            latest_model_path = model_resolver.get_latest_model_path()
            latest_transformer_path = model_resolver.get_latest_transformer_path()
            latest_encoder_path = model_resolver.get_latest_target_encoder_path()
            logging.info(f"Paths {latest_model_path} {latest_transformer_path}")

            predict_pipeline = PredictPipeline(latest_model_path, latest_transformer_path,latest_encoder_path)
            results = predict_pipeline.predict(pred_df)
            
            results_list = results.tolist()
            return render_template('home.html', results=results_list)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    #app.run(host="0.0.0.0", debug=True)
    app.run(host="0.0.0.0",port = 8080)
