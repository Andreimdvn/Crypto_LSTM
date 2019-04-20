import json
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from utils.display_functions import display_model_train_history


class LstmModel:
    DEFAULT_FIRST_LAYER_UNITS = 4
    DEFAULT_EPOCHS_NUMBER = 100

    def __init__(self, ):
        self.model = None
        self.history = None

    def load_from_file(self, file_path):
        self.model = keras.models.load_model(file_path)

    def save_model(self, model_config_file):
        print("Saving model to file {}".format(model_config_file))
        self.model.save(model_config_file)

    def init_model(self, first_layer_units=DEFAULT_FIRST_LAYER_UNITS):
        model = Sequential()
        model.add(LSTM(units=first_layer_units, activation='sigmoid', input_shape=(None, 1)))
        # model.add(LSTM(units=self.first_layer_units*2, activation='sigmoid', input_shape=(None, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, x_train, y_train, epochs=DEFAULT_EPOCHS_NUMBER):
        self.history = self.model.fit(x_train, y_train, batch_size=5, epochs=epochs)
        keras.utils.print_summary(self.model)
        display_model_train_history(self.history, block=False)

    def test_model(self, x_test):
        return self.model.predict(x_test)

    def save_history_to_file(self, history_file):
        out_file = open(history_file, 'wb')
        pickle.dump(self.history, out_file)
