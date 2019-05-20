import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

from utils.display_functions import display_model_train_history


class LstmModel:
    LSTM_UNITS = 30
    DEFAULT_EPOCHS_NUMBER = 100
    DEFAULT_BATCH_SIZE = 64

    def __init__(self, ):
        self.model = None
        self.history = None

    def load_from_file(self, file_path):
        self.model = keras.models.load_model(file_path)

    def save_model(self, model_config_file):
        print("Saving model to file {}".format(model_config_file))
        self.model.save(model_config_file)

    def init_model(self, lstm_ouput_size=LSTM_UNITS):
        model = Sequential()
        model.add(LSTM(units=lstm_ouput_size, activation='sigmoid', input_shape=(60, 1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=lstm_ouput_size, activation='sigmoid'))
        model.add(Dropout(0.2))
        # model.add(LSTM(units=self.first_layer_units*2, activation='sigmoid', input_shape=(None, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(model.summary())
        self.model = model

    def train_model(self, x_train, y_train, epochs=DEFAULT_EPOCHS_NUMBER, batch_size=DEFAULT_BATCH_SIZE):
        self.history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        keras.utils.print_summary(self.model)
        display_model_train_history(self.history, block=False)

    def test_model(self, x_test):
        return self.model.predict(x_test)

    def evaluate_model(self, x_test, y_test):
        return self.model.metrics_names, self.model.evaluate(x_test, y_test)

    def save_history_to_file(self, history_file):
        out_file = open(history_file, 'wb')
        pickle.dump(self.history, out_file)
