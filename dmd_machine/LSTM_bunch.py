#Train all codes together---LSTM bounch Model
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model, Sequential, Model
import numpy as np

class LSTM_bunch(object):
    # Define the LSTM model
    def build_lstm_model_bunch(self,timesteps, latent_dim,units_LSTM):
        lstm_model=Sequential()
        lstm_model.add(LSTM(units_LSTM, activation='tanh',input_shape=(timesteps, latent_dim)))
        lstm_model.add(RepeatVector(timesteps))
        lstm_model.add(LSTM(units_LSTM, activation='tanh',return_sequences=True))
        lstm_model.add(TimeDistributed(Dense(latent_dim)))
        return lstm_model
    def lstm_inputs(self,y_data):
        #the encoded output data without the last time series state.
        return y_data[:-1]
    def lstm_targets(self,y_data):
        #the encoded output data without the first time series state.
        return y_data[1:]
    def extrapolate_lstm(self,lstm_model, encoded_data, future_steps):
        future_latent = []
        current_input = encoded_data
        for _ in range(future_steps):
            # Predict the next latent space
            next_latent = lstm_model.predict(current_input)
            future_latent.append(next_latent)
            current_input=next_latent
        future_latent=np.concatenate(future_latent,axis=1)
        return np.array(future_latent).squeeze()