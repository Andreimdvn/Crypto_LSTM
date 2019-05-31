import argparse
import numpy as np

from data_loading import data_loader_factory
from model.LSTM_model import LstmModel
from utils import defaults
from utils.display_functions import visualize_results
from utils.np_functions import get_price_series_from_start_price_and_percentage


def main(csv_data_file, model_file, days_to_predict, consecutive_predictions, percentage_normalizer, sequence_length):
    data_loader = data_loader_factory.get_data_loader(csv_data_file, days_to_predict, percentage_normalizer,
                                                      sequence_length)
    lstm_model = LstmModel()
    lstm_model.load_from_file(model_file)

    if not consecutive_predictions:
        y_predicted = lstm_model.test_model(data_loader.x_test)
        print(lstm_model.evaluate_model(data_loader.x_test, data_loader.y_test))
    else:
        day_input_to_predict = data_loader.x_test[0]
        print(day_input_to_predict)
        print(day_input_to_predict.shape)
        y_predicted = []
        for _ in range(days_to_predict):
            day_input_to_predict = np.reshape(day_input_to_predict, (1, day_input_to_predict.shape[0], 1))
            new_prediction = lstm_model.test_model(day_input_to_predict)
            # print(day_input_to_predict[0][-1][0] > new_prediction[0][0], day_input_to_predict[0][-1][0], new_prediction[0][0])
            y_predicted.append(new_prediction[0])
            day_input_to_predict = np.append(day_input_to_predict[0][1:], new_prediction)

    actual = data_loader.reverse_min_max(data_loader.y_test)
    predicted = data_loader.reverse_min_max(y_predicted)
    print(actual)
    print(predicted)
    if percentage_normalizer:
        print('actual prices from data loader: ', data_loader.price_values[-20:])
        actual_price = get_price_series_from_start_price_and_percentage(
            data_loader.price_values[-data_loader.sequence_length - 1], actual)
        predicted_price = get_price_series_from_start_price_and_percentage(
            data_loader.price_values[-data_loader.sequence_length - 1], predicted)
        previous_price = data_loader.data['price(USD)'].values[-280:-180]
        previous_price = np.reshape(previous_price, (len(previous_price), 1))
        actual = np.concatenate((previous_price, actual))
        predicted = np.concatenate((previous_price, predicted))
        visualize_results((actual_price, predicted_price), labels=('actual BTC price', 'predicted BTC price'))
        visualize_results((actual, predicted), labels=('actual BTC percentage change',
                                                       'predicted BTC percentage change'))
    else:
        previous_price = data_loader.reverse_min_max(data_loader.y_train)[-100:]
        actual = np.concatenate((previous_price, actual))
        predicted = np.concatenate((previous_price, predicted))
        visualize_results((actual, predicted), labels=('actual BTC price', 'predicted BTC price'))


def init_arg_parser():
    parser = argparse.ArgumentParser(description="Describe model and test it")
    parser.add_argument('-f', '--file_csv', dest='csv_data_file', help='Data file in csv format', type=str,
                        required=True)
    parser.add_argument('-m', '--model_file', dest='model_file', help='LSTM Keras model export file', type=str, required=True)
    parser.add_argument('-d', '--days_to_predict', dest='days_to_predict',
                        help='Days to predict. Training set = last number of days',
                        type=int, default=defaults.DEFAULT_DAYS_TO_PREDICT)
    parser.add_argument('-c', '--consecutive_prediction', dest='consecutive_prediction',
                        help='Will predict based on previous predicted price not on real previous price.',
                        type=bool, default=False)
    parser.add_argument('-p', '--percentage_prediction', dest='percentage',
                        help='Will convert prices to percentage change', default=False, action='store_true')
    parser.add_argument('-s', '--sequence', dest='sequence_length', help='number of timestamps used for prediction',
                        type=int, default=defaults.DEFAULT_SEQUENCE_LENGTH)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg_parser()
    main(args.csv_data_file, args.model_file, int(args.days_to_predict), args.consecutive_prediction,
         args.percentage, args.sequence_length)
