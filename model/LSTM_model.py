import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from tensorflow.python.lib.io import file_io

from utils.display_functions import display_model_train_history


class LstmModel:
    def __init__(self):
        self.model = None
        self.history = None

    def load_from_file(self, file_path):
        self.model = keras.models.load_model(file_path)

    def save_model(self, model_config_file):
        print("Saving model to file {}".format(model_config_file))
        self.model.save(model_config_file)

    def init_model(self, lstm_ouput_size, features):
        model = Sequential()
        model.add(LSTM(units=lstm_ouput_size, activation='tanh', input_shape=(None, features), return_sequences=True,
                       dropout=0.2))
        model.add(Dropout(0.15))
        model.add(LSTM(units=lstm_ouput_size, activation='tanh', return_sequences=True))
        model.add(Dropout(0.15))
        model.add(LSTM(units=lstm_ouput_size, activation='tanh', input_shape=(None, 1)))
        # model.add(Dense(units=lstm_ouput_size))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(model.summary())
        self.model = model

    def train_model(self, x_train, y_train, epochs, batch_size):
        # checkpoint = ModelCheckpoint(filepath=checkpoint_file_prefix + '_checkpoint-{epoch:02d}-{loss:.2f}.hdf5',
        #                              period=self.CHECKPOINT_DUMP_MODEL, verbose=1)

        self.history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)  # validation_split=0.1)
        keras.utils.print_summary(self.model)
        # display_model_train_history(self.history, block=False)

    def test_model(self, x_test):
        return self.model.predict(x_test)

    def evaluate_model(self, x_test, y_test):
        return self.model.metrics_names, self.model.evaluate(x_test, y_test)

    def save_history_to_file(self, history_file):
        if not isinstance(history_file, file_io.FileIO):
            history_file = open(history_file, 'wb')
        pickle.dump(self.history, history_file)
