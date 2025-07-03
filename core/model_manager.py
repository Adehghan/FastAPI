
from ast import mod
import csv
from datetime import datetime
import os
import joblib

import keras
from core.config import MODEL_DIR, logger, MODEL_REGISTRY_FILE
from keras import Model as KerasModel
from sklearn.base import BaseEstimator

from core.enum import ModelTypeEnum

# def save_model(model, accuracy):
#     version = datetime.now().strftime("%Y%m%d_%H%M%S")
#     model_path= os.path.join(MODEL_DIR, f'cnn_model_{accuracy:4f}_{version}.h5')
#     model.save(model_path.replace('.h5', '.keras')) #model.save('my_model.keras')
#     logger.info(f"Model saved to {model_path}")

def save_model(model, model_name, accuracy):
    try:
        # Generate a timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the model file path
        file_path = os.path.join(MODEL_DIR, f"{model_name}_{accuracy}_{timestamp}.h5")
        
        # Save the model
        if isinstance(model, KerasModel):
            model.save(file_path)  # Keras save
        elif isinstance(model, BaseEstimator):
            joblib.dump(model, file_path)  # Sklearn save

        # model.save(file_path)
        logger.info(f"Model saved to {file_path}")
        #print(f"Model saved to {file_path}")
        
        return file_path  # Ensure this is being returned
    except Exception as e:
        print(f"Error saving model: {e}")
        logger.info(f"Error saving model: {e}")
        return None  # If there's an error, return Non


def register_model(model_path, accuracy, status='production'):
    try:
        # Open the model registry CSV file (not the model file itself)
        registry_file = MODEL_REGISTRY_FILE  # This should be a path to your model registry CSV file
        if not os.path.exists(registry_file):
            # Create the registry file if it doesn't exist
            with open(registry_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Accuracy', 'Model Path', 'Status'])  # Write header row

        # Add the model details to the registry file
        with open(registry_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now(), accuracy, model_path, status])
        
        logger.info(f"Model {model_path} registered successfully!")
    except Exception as e:
        logger.error(f"Error registering model: {e}")


def load_model(file_path):
    model = keras.models.load_model(file_path)
    return model


def get_latest_model():
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    model_files.sort(reverse=True)
    latest_model_path = os.path.join(MODEL_DIR, model_files[0]) if model_files else None
    return latest_model_path


def get_latest_model(model_type : ModelTypeEnum):    
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5') and f.startswith(model_type.name)]
    model_files.sort(reverse=True)
    latest_model_path = os.path.join(MODEL_DIR, model_files[0]) if model_files else None
    return latest_model_path
