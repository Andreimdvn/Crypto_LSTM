import argparse
import numpy as np

from data_loading import data_loader_factory
from model.LSTM_model import LstmModel
from utils import defaults
from utils.display_functions import visualize_results, display_plots_with_single_prediction, display_confusion_matrix, \
    plot_result_lines
from utils.metrics import get_binary_accuracy_from_price_prediction, get_confusion_matrix_f1score_for_price_prediction, \
    get_binary_accuracy_from_percentage_prediction, get_confusion_matrix_f1score_for_percentage_prediction, \
    get_mse_series
from utils.np_functions import get_price_series_from_start_price_and_percentage, \
    get_price_series_from_prices_and_percentages


def main(csv_data_file, model_file, days_to_predict, consecutive_predictions, percentage_normalizer, sequence_length,
         log_return):
    data_loader = data_loader_factory.get_data_loader(csv_data_file, days_to_predict, percentage_normalizer,
                                                      sequence_length, log_return)
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
        y_predicted = np.array(y_predicted)
    actual = data_loader.reverse_min_max(np.reshape(data_loader.y_test, (len(data_loader.y_test), 1)))
    predicted = data_loader.reverse_min_max(y_predicted)
    print('y_test shape', data_loader.y_test.shape)
    print('y_pred shape', y_predicted.shape)
    print('Actual: ', actual)
    print('Predicted: ', predicted)
    print('positive actual: ', len(actual[actual > 0]))
    print('positive predicted: ', len(predicted[predicted > 0]))

    if percentage_normalizer:
        actual_price = get_price_series_from_start_price_and_percentage(
            data_loader.price_values[-data_loader.sequence_length - 1], actual)
        predicted_price = get_price_series_from_prices_and_percentages(data_loader.price_values[-data_loader.sequence_length-1:], predicted)
        visualize_results((actual_price, predicted_price), labels=('actual BTC price', 'predicted BTC price'), block=False)
        visualize_results((actual, predicted), labels=('actual BTC percentage change',
                                                       'predicted BTC percentage change'), block=False)
        print("Binary accuracy: ", get_binary_accuracy_from_percentage_prediction(actual, predicted))
        confusion_matrix, f1score = get_confusion_matrix_f1score_for_percentage_prediction(actual, predicted)
        display_confusion_matrix(confusion_matrix)
        print("F1 score: {}".format(f1score))

        actual = actual_price
        predicted = predicted_price
        print("Actual price: ", actual)
        print("Predicted price: ", predicted)
    else:
        plot_result_lines(actual, predicted, block=False)
        previous_price = data_loader.reverse_min_max(np.reshape(data_loader.y_train, (len(data_loader.y_train), 1)))[-100:]
        actual_price = np.concatenate((previous_price, actual))
        predicted_price = np.concatenate((previous_price, predicted))
        visualize_results((actual_price, predicted_price), labels=('actual BTC price', 'predicted BTC price'), block=False)
        mse_series = get_mse_series(data_loader.y_test, y_predicted)
        visualize_results((mse_series,), ('Mean Squared Error',), title='Testing error')

    if not consecutive_predictions:  # plot sample of prediction vs actual in multiple subplots
        if not percentage_normalizer:
            print('x_test shape', data_loader.x_test.shape)
            reshaped_x_test = np.reshape(data_loader.x_test, (len(data_loader.x_test), data_loader.x_test.shape[1] *
                                                              data_loader.x_test.shape[2]))
            actual_price_input = data_loader.reverse_min_max(reshaped_x_test)
            actual_price_input = np.reshape(actual_price_input, data_loader.x_test.shape)
            print("Binary Accuracy: {}".format(get_binary_accuracy_from_price_prediction(actual_price_input[:, :, 0],
                                                                                         actual, predicted)))
            confusion_matrix, f1score = get_confusion_matrix_f1score_for_price_prediction(
                actual_price_input, actual, predicted)
            display_confusion_matrix(confusion_matrix)
            print("F1 score: {}".format(f1score))

            display_plots_with_single_prediction(actual_price_input, actual, predicted)
    input()


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
    parser.add_argument('-L', '--log_return', dest='log_return', help='Will convert prices to log return',
                        default=False, action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg_parser()
    main(args.csv_data_file, args.model_file, int(args.days_to_predict), args.consecutive_prediction,
         args.percentage_prediction, args.sequence_length, args.log_return)
