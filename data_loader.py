import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.display_functions import print_shape_describe_head, print_shape


class BitcoinDataLoader:
    def __init__(self, csv_path, test_set_size, sequence_length):
        """
        :param csv_path: path to data csv format
        :param test_set_size: days to predict
        :param sequence_length: previous days used in prediction
        """
        self.__csv_path = csv_path
        self.__days_to_predict = test_set_size
        self.__all_data_grouped_by_date = self.__read_and_group_by_date()
        self.__train_sequence_date_grouped, self.__test_sequence_date_grouped = self.__split_train_test_data()
        self.__min_max_scaler = MinMaxScaler()
        self.x_train, self.y_train, self.x_test, self.y_test = self.__normalize_data()
        self.log_data_shapes()

    def __read_and_group_by_date(self):
        df = pd.read_csv(self.__csv_path)
        # fill NaN`s with previous price
        df["Weighted_Price"].fillna(method='ffill', inplace=True)
        print_shape_describe_head(df, "Loaded csv and filled Nan")

        df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
        df_grouped_datetime = df.groupby('date')["Weighted_Price"].mean()

        print_shape_describe_head(df_grouped_datetime, "grouped by datetime:")

        return df_grouped_datetime

    def __split_train_test_data(self):
        df_train = self.__all_data_grouped_by_date[:len(self.__all_data_grouped_by_date) - self.__days_to_predict]
        df_test = self.__all_data_grouped_by_date[len(self.__all_data_grouped_by_date) - self.__days_to_predict:]

        print_shape_describe_head(df_train, "Train data")
        print_shape_describe_head(df_test, "Test data")

        return df_train, df_test

    def __normalize_data(self):
        training_set = self.__train_sequence_date_grouped.values
        training_set = np.reshape(training_set, (len(training_set), 1))
        training_set = self.__min_max_scaler.fit_transform(training_set)
        x_train = training_set[0: len(training_set) - 1]
        y_train = training_set[1: len(training_set)]
        x_train = np.reshape(x_train, (len(x_train), 1, 1))

        test_set = self.__test_sequence_date_grouped.values
        test_set = np.reshape(test_set, (len(test_set), 1))
        test_set = self.__min_max_scaler.transform(test_set)
        x_test = test_set[0: len(test_set) - 1]
        y_test = test_set[1: len(test_set)]
        x_test = np.reshape(x_test, (len(x_test), 1, 1))

        return x_train, y_train, x_test, y_test

    def reverse_min_max(self, values):
        return self.__min_max_scaler.inverse_transform(values)

    def log_data_shapes(self):
        print("Data loaded:")
        print_shape(self.x_train, "x_train")
        print_shape(self.y_train, "y_train")
        print_shape(self.x_test, "x_test")
        print_shape(self.y_test, "y_test")
