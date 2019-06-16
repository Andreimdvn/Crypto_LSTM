import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.display_functions import print_shape


class CryptoDataLoader:
    """
    Used to load data for any cryptocurrency
    Using https://datahub.io/cryptocurrency/
    Can load from URL or from local CSV file
    Produces train and test arrays
    """
    def __init__(self, csv_source, days_to_predict, sequence_length, use_percentage, relative_price_change,
                 columns=None):
        self.sequence_length = sequence_length
        self.__days_to_predict = days_to_predict
        self.data = self.__load_data(csv_source)
        print("Data Loaded: {}".format(self.data.describe()))
        self.data = self.data.dropna()
        self.data = self.__filter_columns(columns)
        self.price_values = np.copy(self.data['price(USD)'].values)
        self.features = len(self.data.columns)
        self.__min_max_scaler = MinMaxScaler()
        if use_percentage:
            self.__transform_price_to_percentage()
        if relative_price_change:
            self.x_train, self.y_train, self.x_test, self.y_test = self.__create_relative_price_change_sequences()
        else:
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

    def __create_sequences(self):
        # apply min max scaler
        train = self.data['price(USD)'].values[:-self.__days_to_predict]
        train = self.__min_max_scaler.fit_transform(np.reshape(train, (len(train), 1)))
        test = self.data['price(USD)'].values[-self.__days_to_predict:]
        test = self.__min_max_scaler.transform(np.reshape(test, (len(test), 1)))

        input_sequence = []
        output_sequence = []
        used_values = np.concatenate((train, test))
        for start_day in range(len(used_values) - self.sequence_length - 1):
            input_sequence.append(used_values[start_day: start_day + self.sequence_length])
            output_sequence.append(used_values[start_day + self.sequence_length])

        np_input = np.array(input_sequence)
        np_output = np.array(output_sequence)

        x_train = np_input[:-self.__days_to_predict]
        x_test = np_input[-self.__days_to_predict:]
        y_train = np_output[:-self.__days_to_predict]
        y_test = np_output[-self.__days_to_predict:]

        return x_train, y_train, x_test, y_test

    def __filter_columns(self, columns):
        if columns is None:
            columns = ['price(USD)']
        return self.data[columns]

    def __transform_price_to_percentage(self):
        a = self.data['price(USD)'].values
        b = np.concatenate((np.array([a[0]]), a[:-1]))
        percentages = a/b - 1
        print("Created percentages from prices:", a[-10:], b[-10:], percentages[-10:])
        self.data['price(USD)'] = percentages

    def __create_relative_price_change_sequences(self):
        values = self.data['price(USD)'].values
        sequences = [values[i: i + self.sequence_length + 1] / values[i-1] - 1
                     for i in range(1, len(values) - self.sequence_length)]
        input_sequences = [seq[:-1] for seq in sequences]
        output_sequences = [seq[-1] for seq in sequences]


