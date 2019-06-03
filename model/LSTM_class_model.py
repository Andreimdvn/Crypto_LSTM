import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam

from utils.gcloud_utils import copy_file_to_gcloud


class LstmClassModel:
    def __init__(self):
        self.model = None
        self.history = None

    def loadl_from_file(self, file_path):
        self.model = keras.models.load_model(file_path)

    def save_model(self, model_config_file):
        print("Saving model to file {}".format(model_config_file))
        self.model.save(model_config_file)

    def init_model(self, lstm_units, number_of_layers, dropout_rate, features, learning_rate, classes):
        model = Sequential()
        if number_of_layers == 1:
            model.add(LSTM(units=lstm_units, activation='tanh', input_shape=(None, features)))
            model.add(Dropout(dropout_rate))
        elif number_of_layers == 2:
            model.add(LSTM(units=lstm_units, activation='tanh', input_shape=(None, features), return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(LSTM(units=lstm_units, activation='tanh'))
            model.add(Dropout(dropout_rate))
        elif number_of_layers == 3:
            model.add(LSTM(units=lstm_units, activation='tanh', input_shape=(None, features), return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(LSTM(units=lstm_units, activation='tanh', return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(LSTM(units=lstm_units, activation='tanh'))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        print(model.summary())
        self.model = model

    def train_model(self, x_train, y_train, epochs, batch_size, use_early_stop=False):
        # checkpoint = ModelCheckpoint(filepath=checkpoint_file_prefix + '_checkpoint-{epoch:02d}-{loss:.2f}.hdf5',
        #                              period=self.CHECKPOINT_DUMP_MODEL, verbose=1)
        if use_early_stop:
            es = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto',
                                               restore_best_weights=True)
            self.history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,  validation_split=0.1,
                                          callbacks=[es], shuffle=True)
        else:
            self.history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                                          shuffle=True)
        keras.utils.print_summary(self.model)

    def test_model(self, x_test):
        return self.model.predict(x_test)

    def evaluate_model(self, x_test, y_test):
        return self.model.metrics_names, self.model.evaluate(x_test, y_test)

    def save_history(self, history_file):
        print("Saving history to file {}".format(history_file))
        history_file = open(history_file, 'wb')
        pickle.dump(self.history, history_file)

    def save_model_gcloud(self, output_file, job_dir):
        self.save_model(output_file)
        copy_file_to_gcloud(output_file, job_dir, output_file)

    def save_history_gcloud(self, history_output_file, job_dir):
        self.save_history(history_output_file)
        copy_file_to_gcloud(history_output_file, job_dir, history_output_file)
