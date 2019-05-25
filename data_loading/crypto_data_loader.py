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
    def __init__(self, csv_source, days_to_predict, sequence_length, columns=None):
        self.__sequence_length = sequence_length
        self.__days_to_predict = days_to_predict
        self.__data = self.__load_data(csv_source)
        print("Data Loaded: {}".format(self.__data.describe()))
        self.__data = self.__data.dropna()
        self.__data = self.__filter_columns(columns)
        self.features = len(self.__data.columns)
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

    def __create_sequences(self):
        input_sequence = []
        output_sequence = []
        used_values = self.__data.values
        for start_day in range(len(used_values) - self.__sequence_length - 1):
            input_sequence.append(used_values[start_day: start_day + self.__sequence_length])
            output_sequence.append(used_values[start_day + self.__sequence_length])

        np_input = np.array(input_sequence)
        print_shape(np_input, "debug_np_input")
        np_output = np.array(output_sequence)
        print_shape(np_output, "debug_np_output")

        x_train = np_input[:-self.__days_to_predict]
        x_test = np_input[-self.__days_to_predict:]
        y_train = np_output[:-self.__days_to_predict]
        y_test = np_output[-self.__days_to_predict:]

        x_train = np.reshape(x_train, (len(x_train) * self.__sequence_length, self.features))
        x_test = np.reshape(x_test, (len(x_test) * self.__sequence_length, self.features))

        x_train = self.__min_max_scaler.fit_transform(x_train)
        x_test = self.__min_max_scaler.transform(x_test)
        y_train = self.__min_max_scaler.transform(y_train)
        y_test = self.__min_max_scaler.transform(y_test)

        x_train = np.reshape(x_train, (len(x_train) // self.__sequence_length, self.__sequence_length, self.features))
        x_test = np.reshape(x_test, (len(x_test) // self.__sequence_length, self.__sequence_length, self.features))

        return x_train, y_train, x_test, y_test

    def __filter_columns(self, columns):
        if columns is None:
            columns = ['price(USD)']
        return self.__data[columns]
