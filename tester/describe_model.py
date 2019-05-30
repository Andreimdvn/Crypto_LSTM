import pickle
import sys

import keras

from model.LSTM_model import LstmModel
from utils.display_functions import display_model_train_history


def main(model_file_path, history_file_path):
    lstm_model = LstmModel()
    lstm_model.load_from_file(model_file_path)
    keras.utils.print_summary(lstm_model.model)

    history_file_handle = open(history_file_path, 'rb')
    model_fit_history = pickle.load(history_file_handle)
    print("Epochs: {}".format(len(model_fit_history.history['loss'])))
    print("Loss:")
    print(model_fit_history.history['loss'])
    display_model_train_history(model_fit_history)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: py {} MODEL_PATH MODEL_HISTORY_PATH".format(__name__))
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
