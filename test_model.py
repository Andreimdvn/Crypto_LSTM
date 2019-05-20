import argparse
import numpy as np

import DataLoaderFactory
from LSTM_model import LstmModel
from utils.display_functions import visualize_results


DEFAULT_DAYS_TO_PREDICT = 100


def main(csv_data_file, model_file, days_to_predict, consecutive_predictions):
    data_loader = DataLoaderFactory.get_data_loader(csv_data_file, days_to_predict)
    lstm_model = LstmModel()
    lstm_model.load_from_file(model_file)

    actual_price = data_loader.reverse_min_max(data_loader.y_test)

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
            print(day_input_to_predict[0][-1][0] > new_prediction[0][0], day_input_to_predict[0][-1][0], new_prediction[0][0])
            y_predicted.append(new_prediction[0])
            day_input_to_predict = np.append(day_input_to_predict[0][1:], new_prediction)

    predicted_price = data_loader.reverse_min_max(y_predicted)
    visualize_results((actual_price, predicted_price), labels=('actual BTC price', 'predicted BTC price'))


def init_arg_parser():
    parser = argparse.ArgumentParser(description="Describe model and test it")
    parser.add_argument('-f', dest='csv_data_file', help='Data file in csv format', type=str, required=True)
    parser.add_argument('-m', dest='model_file', help='LSTM Keras model export file', type=str, required=True)
    parser.add_argument('-d', dest='days_to_predict', help='Days to predict. Training set = last number of days',
                        type=int, default=DEFAULT_DAYS_TO_PREDICT)
    parser.add_argument('-c', dest='consecutive_prediction', help='Will predict based on previous predicted price not '
                                                                  'on real previous price.', type=bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg_parser()
    main(args.csv_data_file, args.model_file, int(args.days_to_predict), args.consecutive_prediction)
