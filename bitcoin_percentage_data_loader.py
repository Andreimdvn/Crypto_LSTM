import numpy as np
import pandas as pd
from utils.display_functions import print_shape_describe_head, print_shape


class BitcoinPercentageDataLoader:
    def __init__(self, csv_path, test_set_size, sequence_length):
        """
        :param csv_path: path to data csv format
        :param test_set_size: days to predict
        :param sequence_length: previous days used in prediction
        """
        self.__csv_path = csv_path
        self.__days_to_predict = test_set_size
        self.sequence_length = sequence_length
        self.price_values = self.__read_and_group_by_date()
        self.__percentage_changes = self.__transform_price_to_percentage()
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

        return df_grouped_datetime.values

    def log_data_shapes(self):
        print("Data loaded:")
        print_shape(self.x_train, "x_train")
        print_shape(self.y_train, "y_train")
        print_shape(self.x_test, "x_test")
        print_shape(self.y_test, "y_test")

    def __create_sequences(self):
        input_sequence = []
        output_sequence = []
        percent_values = self.__percentage_changes
        for start_day in range(len(percent_values) - self.sequence_length - 1):
            input_sequence.append(percent_values[start_day: start_day + self.sequence_length])
            output_sequence.append(percent_values[start_day + self.sequence_length])

        np_input = np.array(input_sequence)
        print_shape(np_input, "debug_np_input")
        np_output = np.array(output_sequence)
        print_shape(np_output, "debug_np_output")

        np_input = np.reshape(np_input, (len(np_input), self.sequence_length, 1))
        np_output = np.reshape(np_output, (len(np_output), 1))

        x_train = np_input[:-self.__days_to_predict]
        x_test = np_input[-self.__days_to_predict:]
        y_train = np_output[:-self.__days_to_predict]
        y_test = np_output[-self.__days_to_predict:]

        return x_train, y_train, x_test, y_test

    def __transform_price_to_percentage(self):
        a = self.price_values
        b = np.concatenate((np.array([a[0]]), a[:-1]))
        percentages = a/b - 1
        print("Created percentages from prices:", a[-10:], b[-10:], percentages[-10:])
        return percentages
