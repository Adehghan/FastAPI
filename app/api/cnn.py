from gc import callbacks
from sys import exception
from turtle import mode
from fastapi import APIRouter, BackgroundTasks, HTTPException
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
import logging

from core import model_manager
from core.config import MODEL_DIR
# from datetime import datetime
# import os
# from core.config import MODEL_DIR

router = APIRouter(prefix='/model', tags=['Deep Learning Models'])
logger = logging.getLogger(__name__)

result = {'code' : None, 'result' : None}

def train_model():
    global result
    try:
        logger.info("Started training CNN model.")

        # Load data
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        print(X_train.shape, y_train.shape)

        # Normalization
        X_train = X_train.reshape(-1,28,28,1) / 255
        X_test = X_test.reshape(-1,28,28,1) / 255
        print(X_train.shape, y_train.shape)

        # Create Model
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

        model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides=(1,1), input_shape=(28,28,1)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.AvgPool2D(pool_size=(2,2), strides=(2,2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units = 32, activation = 'relu'))
        model.add(keras.layers.Dense(units = 64, activation = 'relu'))
        model.add(keras.layers.Dense(units = 10, activation = 'softmax'))
        
        model.summary()

        # Compile
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Training
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=30)        
        history = model.fit(X_train, y_train, batch_size=256, epochs=100, callbacks=early_stop, validation_data=(X_test, y_test))

        # Prediction
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)

        result['code'] = 0
        result['result'] = accuracy

        logger.info(f"Training completed with accuracy: {accuracy:.4f}")

        # Saving Model
        model_path = model_manager.save_model(model, 'cnn_model', accuracy)
        model_manager.register_model(model_path, accuracy)

        # version = datetime.now().strftime("%Y%m%d_%H%M%S")
        # model_path= os.path.join(MODEL_DIR, f'cnn_model_{accuracy:4f}_{version}.h5')
        # model.save(model_path.replace('.h5', '.keras')) #model.save('my_model.keras')
        # logger.info(f"Model saved to {model_path}")

        return accuracy
    except Exception as e:
        logger.error("Training failed due to an exception", exc_info=True)
        #HTTPException(status_code=500, detail=f"Error while checking training status: {str(e)}")
        result['code'] = -1
        result['result'] = None
        return None
    
def train_cnn_dropout():
    logger.info("Started training CNN model.")
    
    try:
        # Load Data
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        print(X_train.shape, y_train.shape)

        # Normalization
        X_train = X_train.reshape(-1,28,28,1).astype('float32') / 255
        X_test = X_test.reshape(-1,28,28,1).astype('float32') / 255
        print(X_train.shape, X_test.shape)

        # Model Generation
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

        model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units = 32, activation = 'relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(units = 64, activation = 'relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(units = 10, activation = 'softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        # Training
        model.fit(X_train, y_train, batch_size=256, epochs = 10, validation_data= (X_test, y_test))

        # Prediction
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)

        result['code'] = 0
        result['result'] = accuracy

        logger.info(f"Training completed with accuracy: {accuracy:.4f}")

        # Saving Model
        model_path = model_manager.save_model(model, 'cnn_dropout', accuracy)
        model_manager.register_model(model_path, accuracy)

        return accuracy

    except exception as e:
        logger.error("Training failed due to an exception", exc_info=True)

        result['code'] = -1
        result['result'] = None
        return None
    
@router.get('/cnn')
def create_cnn_model():
    accuracy = train_model()

    return{
        'error_code': 0 if accuracy != None else -1,
        'message' : 'Ok' if accuracy != None else 'An error occured while training.',
        'data' : accuracy
    }

@router.get('/cnn_async')
def create_cnn_model_async(back_ground_task : BackgroundTasks):
    back_ground_task.add_task(train_model)

    return{
        'error_code': result['code'],
        'message' : 'Ok' if result['code'] == 0 else 'An error occured while training.',
        'data' : result['result']
    }

@router.get('/cnn_status')
def check_training_status():
    return {
        'error_code': result['code'],
        'message': 'Training completed' if result['code'] == 0 else 'Training in progress or failed',
        'data': result['result']
    }

@router.get('/cnn_latest_model')
def load_cnn_latest_model():
    latest_model_path = model_manager.get_latest_model()
    if latest_model_path:
        model = model_manager.load_model(latest_model_path)
        return {"message": f"Loaded model from {latest_model_path}"}
    else:
        return {"message": "No models found!"}    

@router.post('/cnn_dropout')
def cnn_dropout(background_task : BackgroundTasks):
    background_task.add_task(train_cnn_dropout)

    return{
        'error_code': result['code'],
        'message' : 'Ok' if result['code'] == 0 else 'An error occured while training.',
        'data' : result['result']
    }