import argparse
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score

from data_loading import data_loader_factory
from model.LSTM_class_model import LstmClassModel
from utils import defaults
from utils.display_functions import display_confusion_matrix


def main(csv_data_file, model_file, days_to_predict, consecutive_predictions, percentage_normalizer, sequence_length):
    data_loader = data_loader_factory.get_data_loader(csv_data_file, days_to_predict, percentage_normalizer,
                                                      sequence_length, relative_price_change=False,
                                                      classification_output=True)
    lstm_model = LstmClassModel()
    lstm_model.load_from_file(model_file)
    print(lstm_model.model.summary())
    if not consecutive_predictions:
        y_predicted = lstm_model.test_model(data_loader.x_test)
        print(y_predicted)
        print(lstm_model.evaluate_model(data_loader.x_test, data_loader.y_test))
        predicted_class = []
        for predicted in y_predicted:
            if predicted < 0.5:
                predicted_class.append(0)
            else:
                predicted_class.append(1)
        predicted_class = np.array(predicted_class)
        conf_matrix = confusion_matrix(data_loader.y_test, predicted_class).ravel()
        display_confusion_matrix(conf_matrix)
        f1score = f1_score(data_loader.y_test, predicted_class)
        print("F1 score: {}".format(f1score))


def init_arg_parser():
    parser = argparse.ArgumentParser(description="Describe model and test it")
    parser.add_argument('-f', '--file_csv', dest='csv_data_file', help='Data file in csv format',
                        type=str, required=True)
    parser.add_argument('-m', '--model_file', dest='model_file', help='LSTM Keras model export file',
                        type=str, required=True)
    parser.add_argument('-d', '--days_to_predict', dest='days_to_predict',
                        help='Days to predict. Training set = last number of days',
                        type=int, default=defaults.DEFAULT_DAYS_TO_PREDICT)
    parser.add_argument('-c', '--consecutive_prediction', dest='consecutive_prediction',
                        help='Will predict based on previous predicted price not on real previous price.',
                        default=False, action='store_true')
    parser.add_argument('-p', '--percentage_prediction', dest='percentage_prediction',
                        help='Will convert prices to percentage change', default=False, action='store_true')
    parser.add_argument('-s', '--sequence_length', dest='sequence_length',
                        help='number of timestamps used for prediction',
                        type=int, default=defaults.DEFAULT_SEQUENCE_LENGTH)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg_parser()
    main(args.csv_data_file, args.model_file, int(args.days_to_predict), args.consecutive_prediction,
         args.percentage_prediction, args.sequence_length)
