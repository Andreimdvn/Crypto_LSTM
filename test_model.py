import argparse
import sys
import os
import time
import datetime

from LSTM_model import LstmModel
from data_loader import DataLoader
from utils.display_functions import visualize_results


DEFAULT_DAYS_TO_PREDICT = 100


def main(csv_data_file, model_file, days_to_predict):
    data_loader = DataLoader(csv_data_file, test_set_size=days_to_predict)
    lstm_model = LstmModel()
    lstm_model.load_from_file(model_file)

    y_predicted = lstm_model.test_model(data_loader.x_test)

    actual_price = data_loader.reverse_min_max(data_loader.y_test)
    predicted_price = data_loader.reverse_min_max(y_predicted)
    predicted_price = predicted_price[1:]
    visualize_results((actual_price, predicted_price), labels=('actual BTC price', 'predicted BTC price'))


def init_arg_parser():
    parser = argparse.ArgumentParser(description="Describe model and test it")
    parser.add_argument('-f', dest='csv_data_file', help='Data file in csv format', type=str, required=True)
    parser.add_argument('-m', dest='model_file', help='LSTM Keras model export file', type=str, required=True)
    parser.add_argument('-d', dest='days_to_predict', help='Days to predict. Training set = last number of days',
                        type=int, default=DEFAULT_DAYS_TO_PREDICT)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg_parser()
    main(args.csv_data_file, args.model_file, int(args.days_to_predict))
    import pika
    ch = pika.BlockingConnection()
    ch.basic