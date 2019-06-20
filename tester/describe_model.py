import os
import pickle
import sys

import keras

from model.LSTM_model import LstmModel
from utils.display_functions import display_model_train_history_loss, display_model_train_history_acc


def main(model_file_path, history_file_path):
    lstm_model = LstmModel()
    lstm_model.load_from_file(model_file_path)
    keras.utils.print_summary(lstm_model.model)

    history_file_handle = open(history_file_path, 'rb')
    model_fit_history = pickle.load(history_file_handle)
    print("Epochs: {}".format(len(model_fit_history.history['loss'])))
    print("Loss:")
    print(model_fit_history.history['loss'])
    display_model_train_history_loss(model_fit_history)
    if 'acc' in model_fit_history.history:
        display_model_train_history_acc(model_fit_history)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: py {} MODEL_PATH OPTIONAL_MODEL_HISTORY_PATH".format(__name__))
        sys.exit(1)
    model_file = sys.argv[1]
    if len(sys.argv) < 3:
        folder, file = os.path.split(model_file)
        history_file = "{}/history_{}".format(folder, file)
    else:
        history_file = sys.argv[2]
    main(model_file, history_file)
