import argparse
import sys
import os

from utils import defaults
from data_loading import data_loader_factory
from LSTM_model import LstmModel
from utils.display_functions import visualize_results
from utils.np_functions import get_price_series_from_start_price_and_percentage


def main(csv_data_file, days_to_predict, epochs, batch_size, lstm_units, sequence_length, percentage_normalizer,
         output_file):

    data_loader = data_loader_factory.get_data_loader(csv_data_file, days_to_predict, percentage_normalizer,
                                                      sequence_length)
    lstm_model = LstmModel()
    lstm_model.init_model(lstm_units, data_loader.features)
    # if continue_training_model:
    #     print("Will continue training from {}".format(continue_training_model))
    #     lstm_model.load_from_file(continue_training_model)
    #     output_model_file = "continued_" + output_model_file

    lstm_model.train_model(data_loader.x_train, data_loader.y_train, epochs, batch_size)
    lstm_model.save_model(output_file)
    lstm_model.save_history_to_file("history_{}".format(output_file))
    y_predicted = lstm_model.test_model(data_loader.x_test)
    print(lstm_model.evaluate_model(data_loader.x_test, data_loader.y_test))

    if percentage_normalizer:
        actual_price = get_price_series_from_start_price_and_percentage(
            data_loader.get_last_training_price(), data_loader.y_test)
        predicted_price = get_price_series_from_start_price_and_percentage(
            data_loader.get_last_training_price(), y_predicted)
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
                        type=int, default=defaults.DEFAULT_DAYS_TO_PREDICT)
    parser.add_argument('-e', dest='epochs', help='Number of epochs used at training', type=int,
                        default=defaults.DEFAULT_EPOCHS_NUMBER)
    parser.add_argument('-b', dest='batch_size', help='Batch size', type=int, default=defaults.DEFAULT_BATCH_SIZE)
    parser.add_argument('-u', dest='lstm_units', help='size of the output of the LSTM', type=int,
                        default=defaults.LSTM_UNITS)
    parser.add_argument('-s', dest='sequence_length', help='number of timestamps used for prediction', type=int,
                        default=defaults.DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument('-p', dest='percentage', help='Will convert prices to percentage change', type=bool,
                        default=False)
    parser.add_argument('-o', dest='output_file', help='{prefix}_epochs_batch_sequence_predictdays_LSTMunits', type=str,
                        default='TBD')

    return parser.parse_args()


def setup_output_file_name(args):
    output_file_name = "{}_{}epochs_{}batch_{}sequence_{}predictdays_{}LSTMunits.cfg".\
        format(args.output_file, args.epochs, args.batch_size, args.sequence_length, args.days_to_predict,
               args.lstm_units)
    if os.path.exists(output_file_name):
        print("File with name {} already exists!".format(output_file_name))
        sys.exit(1)

    args.output_file = output_file_name


if __name__ == "__main__":
    args = init_arg_parser()
    setup_output_file_name(args)
    main(args.csv_data_file, args.days_to_predict, int(args.epochs), int(args.batch_size), int(args.lstm_units),
         int(args.sequence_length), args.percentage, args.output_file)
