#Train each code using a single LSTM model---LSTM-individual Model
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model, Sequential, Model
import numpy as np
import pandas as pd

class LSTM_individual(object):
    # Define the LSTM model
    def build_lstm_model_individual(self,seq_length, latent_dim,units_LSTM):
        lstm_model=Sequential()
        lstm_model.add(LSTM(units_LSTM, return_sequences=True, activation='tanh',input_shape=(seq_length, latent_dim)))
        lstm_model.add(LSTM(units_LSTM, activation='tanh'))
        lstm_model.add(Dense(latent_dim))
        return lstm_model
    def create_sequences(self,data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)   
            
    def extrapolate_lstm(self,lstm_model, last_sequence, future_steps):
        future_predictions = []
    
        for _ in range(future_steps):
            # Predict the next step
            next_prediction = lstm_model.predict(last_sequence[np.newaxis, :])
            future_predictions.append(next_prediction[0])
        
            # Update last_sequence to include the newly predicted steps
            last_sequence = np.vstack([last_sequence[1:], next_prediction])
        return np.array(future_predictions)