import argparse
import os
import time
import gc

import numpy as np
import pandas as pd

from data_loading import data_loader_factory
from model.LSTM_model import LstmModel
from utils import defaults
from utils.metrics import get_binary_accuracy_from_price_prediction, get_confusion_matrix_f1score_for_price_prediction


def get_model__path_form_dir_and_prefix(dir, prefix, sequence, lstm_units, batch_size):
    """
    Searches the dir for a file that starts with given prefix and returns the full path of that file
    """
    for root, _, files in os.walk(dir):
        for file in files:
            if file.startswith(prefix) and sequence in file and lstm_units in file and batch_size in file:
                return os.path.join(root, file)


def evaluate_model(model_file, csv_data_file, days_to_predict, sequence_length, multiple_features):
    start_loading = time.time()
    data_loader = data_loader_factory.get_data_loader(csv_data_file, days_to_predict, percentage_normalizer=False,
                                                      sequence_length=sequence_length, log_return=False,
                                                      multiple_features=multiple_features)
    end_loading = time.time()
    print('Data loaded in {}s'.format(end_loading - start_loading))
    start_loading = time.time()
    lstm_model = LstmModel()
    lstm_model.load_from_file(model_file)
    end_loading = time.time()
    print('Model loaded in {}s'.format(end_loading - start_loading))

    y_predicted = lstm_model.test_model(data_loader.x_test)
    actual = data_loader.reverse_min_max_y(np.reshape(data_loader.y_test, (len(data_loader.y_test), 1)))
    predicted = data_loader.reverse_min_max_y(y_predicted)
    reshaped_x_test = np.reshape(data_loader.x_test, (
        data_loader.x_test.shape[0] * data_loader.x_test.shape[1], data_loader.x_test.shape[2]))
    actual_price_input = data_loader.reverse_min_max(reshaped_x_test)
    actual_price_input = np.reshape(actual_price_input, data_loader.x_test.shape)[:, :, 0]
    acc = get_binary_accuracy_from_price_prediction(actual_price_input, actual, predicted)
    confusion_matrix, f1score = get_confusion_matrix_f1score_for_price_prediction(actual_price_input, actual, predicted)
    lstm_model.delete()

    return acc, confusion_matrix, f1score


def main(csv_data_file, tunning_results, days_to_predict, multiple_features):
    tunning_data = pd.read_csv(tunning_results)
    tunning_data = tunning_data.sort_values('val_loss')
    accuracies = []
    confusion_matrices = []
    f_ones = []
    for idx, model in tunning_data.iterrows():
        model_file = get_model__path_form_dir_and_prefix(os.path.split(tunning_results)[0], model['model_file'],
                                                         "{}sequence".format(model['sequence_length']),
                                                         "{}LSTM".format(model['lstm_units']),
                                                         "{}batch".format(model['batch_size']))
        gc.collect()
        acc, conf_matrix, f1 = evaluate_model(model_file, csv_data_file, days_to_predict, int(model['sequence_length']),
                                              multiple_features)
        accuracies.append(acc)
        confusion_matrices.append("{} {} {} {}".format(*conf_matrix))
        f_ones.append(f1)
        print("Model {}: Acc: {}, Conf_matrix: {}, F1: {}".format(model['model_file'], acc, conf_matrix, f1))

    tunning_data['acc'] = accuracies
    tunning_data['conf_matrix'] = confusion_matrices
    tunning_data['f_one'] = f_ones

    tunning_data.to_csv(tunning_results)
    print("Done. Saved updated results.")


def init_arg_parser():
    parser = argparse.ArgumentParser(description="Describe model and test it")
    parser.add_argument('-f', '--file_csv', dest='csv_data_file', help='Data file in csv format',
                        type=str, required=True)
    parser.add_argument('-t', '--tunning_results', dest='tunning_results',
                        help='Tunning results in the same folder with the models', type=str, required=True)
    parser.add_argument('-d', '--days_to_predict', dest='days_to_predict',
                        help='Days to predict. Training set = last number of days',
                        type=int, default=defaults.DEFAULT_DAYS_TO_PREDICT)
    parser.add_argument('-M', '--multiple_features', dest='multiple_features',
                        help='Use multiple features alongside price', default=False, action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg_parser()
    main(args.csv_data_file, args.tunning_results, int(args.days_to_predict), args.multiple_features)
