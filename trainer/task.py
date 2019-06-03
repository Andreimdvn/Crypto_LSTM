import argparse

from model.LSTM_model import LstmModel
from data_loading import data_loader_factory
from utils import defaults
from utils.display_functions import display_model_train_history
from utils.format_functions import get_output_file_name


def main(csv_data_file, days_to_predict, epochs, batch_size, lstm_units, sequence_length, number_of_layers,
         dropout_rate, learning_rate, percentage_normalizer, output_file, use_early_stop, job_dir):
    output_file = get_output_file_name(output_file, days_to_predict, epochs, batch_size, lstm_units, sequence_length,
                                       number_of_layers, dropout_rate, learning_rate, percentage_normalizer)
    history_output_file = "history_{}".format(output_file)

    data_loader = data_loader_factory.get_data_loader(csv_data_file, days_to_predict, percentage_normalizer,
                                                      sequence_length)
    lstm_model = LstmModel()
    lstm_model.init_model(lstm_units, number_of_layers, dropout_rate, data_loader.features, learning_rate)

    lstm_model.train_model(data_loader.x_train, data_loader.y_train, epochs, batch_size, use_early_stop)
    if job_dir:
        lstm_model.save_model_gcloud(output_file, job_dir)
        lstm_model.save_history_gcloud(history_output_file, job_dir)
    else:
        lstm_model.save_model(output_file)
        lstm_model.save_history(history_output_file)
        # display_model_train_history(lstm_model.history)

    return lstm_model.history


def init_arg_parser():
    parser = argparse.ArgumentParser(description="Train and test a lstm model", fromfile_prefix_chars='@')
    parser.add_argument('-f', '--file_csv', dest='csv_data_file', help='csv file/url', type=str, required=True)
    parser.add_argument('-d', '--days_to_predict', dest='days_to_predict',
                        help='Days to predict. Training set = last number of days',
                        type=int, default=defaults.DEFAULT_DAYS_TO_PREDICT)
    parser.add_argument('-e', '--epochs', dest='epochs', help='Number of epochs used at training',
                        type=int, default=defaults.DEFAULT_EPOCHS_NUMBER)
    parser.add_argument('-b', '--batch_size', dest='batch_size', help='Batch size',
                        type=int, default=defaults.DEFAULT_BATCH_SIZE)
    parser.add_argument('-u', '--lstm_units', dest='lstm_units', help='size of the output of the LSTM',
                        type=int, default=defaults.LSTM_UNITS)
    parser.add_argument('-s', '--sequence_length', dest='sequence_length',
                        help='number of timestamps used for prediction',
                        type=int, default=defaults.DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument('-n', '--number_of_layers', dest='number_of_layers', help='number of lstm layers in the model',
                        type=int, default=defaults.DEFAULT_LSTM_LAYERS)
    parser.add_argument('-r', '--dropout_rate', dest='dropout_rate', help='dropout rate in the dropout layers',
                        type=float, default=defaults.DEFAULT_DROPOUT_RATE)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', help='learning rate of the adam optimizer',
                        type=float, default=defaults.DEFAULT_LEARNING_RATE)
    parser.add_argument('-p', '--percentage_prediction', dest='percentage',
                        help='Will convert prices to percentage change', default=False, action='store_true')
    parser.add_argument('-o', '--output_file', dest='output_file',
                        help='{prefix}_epochs_batch_sequence_predictdays_LSTMunits_layers_drop_lr.cfg', type=str,
                        default='None')
    parser.add_argument('-E', '--early_stop', dest='early_stop',
                        help='Stop the training after 10 epochs if val_loss has not improved', default=False,
                        action='store_true')
    parser.add_argument("-j", '--job-dir', dest='job_dir', help='jobs dir used for gcloud training', required=False,
                        default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg_parser()
    print(args.__dict__)
    main(args.csv_data_file, args.days_to_predict, int(args.epochs), int(args.batch_size), int(args.lstm_units),
         int(args.sequence_length), int(args.number_of_layers), float(args.dropout_rate), float(args.learning_rate),
         args.percentage, args.output_file, args.early_stop, args.job_dir)
