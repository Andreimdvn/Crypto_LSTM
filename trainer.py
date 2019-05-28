import argparse
import sys
import os

from utils import defaults
from data_loading import data_loader_factory
from LSTM_model import LstmModel


def main(csv_data_file, days_to_predict, epochs, batch_size, lstm_units, sequence_length, percentage_normalizer,
         output_file):
    output_file = get_output_file_name(output_file, epochs, batch_size, sequence_length, days_to_predict, lstm_units)
    data_loader = data_loader_factory.get_data_loader(csv_data_file, days_to_predict, percentage_normalizer,
                                                      sequence_length)
    lstm_model = LstmModel()
    lstm_model.init_model(lstm_units, data_loader.features)

    lstm_model.train_model(data_loader.x_train, data_loader.y_train, epochs, batch_size)
    lstm_model.save_model(output_file)
    lstm_model.save_history_to_file("history_{}".format(output_file))

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


def get_output_file_name(output_file, epochs, batch_size, sequence_length, days_to_predict, lstm_units):
    output_file_name = "{}_{}epochs_{}batch_{}sequence_{}predictdays_{}LSTMunits.cfg".\
        format(output_file, epochs, batch_size, sequence_length, days_to_predict, lstm_units)
    if os.path.exists(output_file_name):
        print("File with name {} already exists!".format(output_file_name))
        sys.exit(1)

    return output_file_name


if __name__ == "__main__":
    args = init_arg_parser()
    main(args.csv_data_file, args.days_to_predict, int(args.epochs), int(args.batch_size), int(args.lstm_units),
         int(args.sequence_length), args.percentage, args.output_file)
