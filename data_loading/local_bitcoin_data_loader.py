import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.display_functions import print_shape_describe_head, print_shape


class LocalBitcoinDataLoader:
    def __init__(self, csv_path, test_set_size, sequence_length):
        """
        :param csv_path: path to data csv format
        :param test_set_size: days to predict
        :param sequence_length: previous days used in prediction
        """
        self.__csv_path = csv_path
        self.__days_to_predict = test_set_size
        self.__sequence_length = sequence_length
        self.__all_data_grouped_by_date = self.__read_and_group_by_date()
        self.__min_max_scaler = MinMaxScaler()
        self.x_train, self.y_train, self.x_test, self.y_test = self.__create_sequences()
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

    def reverse_min_max(self, values):
        return self.__min_max_scaler.inverse_transform(values)

    def log_data_shapes(self):
        print("Data loaded:")
        print_shape(self.x_train, "x_train")
        print_shape(self.y_train, "y_train")
        print_shape(self.x_test, "x_test")
        print_shape(self.y_test, "y_test")

    def __create_sequences(self):
        input_sequence = []
        output_sequence = []
        price_values = self.__all_data_grouped_by_date.values
        for start_day in range(len(price_values) - self.__sequence_length - 1):
            input_sequence.append(price_values[start_day: start_day + self.__sequence_length])
            output_sequence.append(price_values[start_day + self.__sequence_length])

        np_input = np.array(input_sequence)
        print_shape(np_input, "debug_np_input")
        np_output = np.array(output_sequence)
        print_shape(np_output, "debug_np_output")

        np_output = np.reshape(np_output, (len(np_output), 1))

        x_train = np_input[:-self.__days_to_predict]
        x_test = np_input[-self.__days_to_predict:]
        y_train = np_output[:-self.__days_to_predict]
        y_test = np_output[-self.__days_to_predict:]

        x_train = np.reshape(x_train, (len(x_train) * self.__sequence_length, 1))
        x_test = np.reshape(x_test, (len(x_test) * self.__sequence_length, 1))

        x_train = self.__min_max_scaler.fit_transform(x_train)
        x_test = self.__min_max_scaler.transform(x_test)
        y_train = self.__min_max_scaler.transform(y_train)
        y_test = self.__min_max_scaler.transform(y_test)

        x_train = np.reshape(x_train, (len(x_train) // self.__sequence_length, self.__sequence_length, 1))
        x_test = np.reshape(x_test, (len(x_test) // self.__sequence_length, self.__sequence_length, 1))

        return x_train, y_train, x_test, y_test
