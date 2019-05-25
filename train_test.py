import argparse
import sys
import os
import time
import datetime

import DataLoaderFactory
from LSTM_model import LstmModel
from utils.display_functions import visualize_results
from utils.np_functions import get_series_from_start_price_and_percentage

DEFAULT_DAYS_TO_PREDICT = 100
DEFAULT_EPOCHS = 100


def get_input_args():
    if len(sys.argv) < 2:
        print("Usage: {} data_file model_file(optional)".format(__file__))
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print("Input path {} does not exist!".format(input_file))
        sys.exit(2)

    model_file = None
    if len(sys.argv) >= 3:
        model_file = sys.argv[2]
        if not os.path.exists(model_file):
            print("Model file path {} does not exist!".format(model_file))
            sys.exit(3)

    return input_file, model_file


def main(csv_data_file, days_to_predict, epochs, output_model_file, percentage_normalizer, continue_training_model):
    if not epochs:
        epochs = LstmModel.DEFAULT_EPOCHS_NUMBER

    data_loader = DataLoaderFactory.get_data_loader(csv_data_file, days_to_predict, percentage_normalizer)
    lstm_model = LstmModel()
    lstm_model.init_model()
    if continue_training_model:
        print("Will continue training from {}".format(continue_training_model))
        lstm_model.load_from_file(continue_training_model)
        output_model_file = "continued_" + output_model_file

    lstm_model.train_model(data_loader.x_train, data_loader.y_train, epochs, checkpoint_file_prefix=output_model_file)
    lstm_model.save_model(output_model_file)
    lstm_model.save_history_to_file("history_{}".format(output_model_file))
    y_predicted = lstm_model.test_model(data_loader.x_test)
    print(lstm_model.evaluate_model(data_loader.x_test, data_loader.y_test))

    if percentage_normalizer:
        actual_price = get_series_from_start_price_and_percentage(
            data_loader.price_values[-data_loader.sequence_length - 1], data_loader.y_test)
        predicted_price = get_series_from_start_price_and_percentage(
            data_loader.price_values[-data_loader.sequence_length - 1], y_predicted)
        visualize_results((actual_price, predicted_price), labels=('actual BTC price', 'predicted BTC price'))
        visualize_results((data_loader.y_test, y_predicted), labels=('actual BTC percentage change',
                                                                     'predicted BTC percentage change'))
    else:
        actual_price = data_loader.reverse_min_max(data_loader.y_test)
        predicted_price = data_loader.reverse_min_max(y_predicted)
        visualize_results((actual_price, predicted_price), labels=('actual BTC price', 'predicted BTC price'))


def init_arg_parser():
    parser = argparse.ArgumentParser(description="Train and test a lstm model")
    parser.add_argument('-f', dest='csv_data_file', help='Data file in csv format', type=str, required=True)
    parser.add_argument('-d', dest='days_to_predict', help='Days to predict. Training set = last number of days',
                        type=int, default=DEFAULT_DAYS_TO_PREDICT)
    parser.add_argument('-e', dest='epochs', help='Number of epochs used at trainig', default=DEFAULT_EPOCHS,
                        required=False)
    current_time = str(datetime.datetime.fromtimestamp(time.time())).replace(':', '_')
    parser.add_argument('-o', dest='output_file', help='Output file to dump model to',
                        default='model_{}.cfg'.format(current_time), required=False)
    parser.add_argument('-p', dest='percentage_normalizer',
                        help='Will convert prices to percentage change and use them as normalization for data [-1:1]',
                        type=bool, default=False)
    parser.add_argument('-c', dest='continue_training_model', help="Continue training the model from this model file",
                        type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg_parser()
    main(args.csv_data_file, args.days_to_predict, int(args.epochs), args.output_file, args.percentage_normalizer,
         args.continue_training_model)
