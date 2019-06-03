import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.display_functions import print_shape, visualize_results


class CryptoDataLoaderClassifier:
    """
    Used to load data for any cryptocurrency
    Using https://datahub.io/cryptocurrency/
    Can load from URL or from local CSV file
    Produces train and test arrays for classification model
    Classes: [1,0,0] = < -1%
             [0,1,0] = change between -1% and 1%
             [0,0,1] = > 1%
    """
    # rate = 0.0001
    # CLASSES = {"down": [1, 0, 0], "none": [0, 1, 0], "up": [0, 0, 1]}
    # CLASS_LABELS = ["down", "none", "up"]
    CLASSES = {"down": [1, 0], "up": [0, 1]}
    CLASS_LABELS = ["down", "up"]

    def __init__(self, csv_source, days_to_predict, sequence_length, use_percentage, columns=None):
        self.sequence_length = sequence_length
        self.__days_to_predict = days_to_predict
        self.data = self.__load_data(csv_source)
        self.data = self.data.dropna()
        self.data = self.__filter_columns(columns)
        print("Data Loaded: {}".format(self.data[20:40]))
        # self.price_values = np.copy(self.data['price(USD)'].values)
        self.features = len(self.data.columns)
        if use_percentage:
            raise Exception("Not implemented!")
        else:
            self.__min_max_scaler = MinMaxScaler()
            self.x_train, self.y_train, self.x_test, self.y_test = self.__create_sequences()
        self.log_data_shapes()

    def __load_data(self, csv_source):
        start = time.time()
        if 'http'in csv_source:
            import datapackage
            package = datapackage.Package(csv_source)
            resources = package.resources
            for resource in resources:
                if resource.tabular:
                    return pd.read_csv(resource.descriptor['path'])
        else:
            return pd.read_csv(csv_source)
        end = time.time()
        print("Loaded data in {} seconds".format(end - start))

    def log_data_shapes(self):
        print("Data loaded:")
        print_shape(self.x_train, "x_train")
        print_shape(self.y_train, "y_train")
        print_shape(self.x_test, "x_test")
        print_shape(self.y_test, "y_test")

    def reverse_min_max(self, values):
        return self.__min_max_scaler.inverse_transform(values)

    def get_class_from_prediction(self, pred):
        return self.CLASS_LABELS[np.argmax(pred)]

    def __create_sequences(self):
        # apply min max scaler
        train_price = self.data['price(USD)'].values[:-self.__days_to_predict]
        train = self.__min_max_scaler.fit_transform(np.reshape(train_price, (len(train_price), 1)))
        test_price = self.data['price(USD)'].values[-self.__days_to_predict:]
        test = self.__min_max_scaler.transform(np.reshape(test_price, (len(test_price), 1)))

        input_sequence = []
        used_values = np.concatenate((train, test))
        output = []
        percentage_change = self.__get_percentage_from_prices()
        for start_day in range(len(used_values) - self.sequence_length - 1):
            input_sequence.append(used_values[start_day: start_day + self.sequence_length])
            percentage = percentage_change[start_day + self.sequence_length]
            if percentage < 0:
                output.append(1)
            else:
                output.append(0)
            # if percentage < -self.rate:
            #     output.append(self.CLASSES["down"])
            # elif percentage > self.rate:
            #     output.append(self.CLASSES["up"])
            # else:
            #     output.append(self.CLASSES["none"])
            # print("Price values: {} percentage_change: {} output: {}".format(
            #     self.data['price(USD)'].values[start_day: start_day + self.sequence_length + 1], input_sequence[-1], output[-1]))

        np_input = np.array(input_sequence)
        x_train = np_input[:-self.__days_to_predict]
        x_test = np_input[-self.__days_to_predict:]

        np_output = np.array(output)
        y_train = np_output[:-self.__days_to_predict]
        y_test = np_output[-self.__days_to_predict:]

        return x_train, y_train, x_test, y_test

    def __filter_columns(self, columns):
        if columns is None:
            columns = ['price(USD)']
        return self.data[columns]

    def __get_percentage_from_prices(self):
        a = self.data['price(USD)'].values
        b = np.concatenate((np.array([a[0]]), a[:-1]))
        percentages = a/b - 1
        print("Created percentages from prices:", a[-10:], b[-10:], percentages[-10:])
        return percentages
