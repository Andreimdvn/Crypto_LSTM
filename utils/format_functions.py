import os
import sys


def get_output_file_name(output_file, days_to_predict, epochs, batch_size, lstm_units, sequence_length,
                         number_of_layers, dropout_rate, learning_rate, percentage_normalizer):
    output_file_name = "{}_{}epochs_{}batch_{}sequence_{}predictdays_{}LSTMunits_{}layers_{}drop_{}lr_{}.cfg".\
        format(output_file, epochs, batch_size, sequence_length, days_to_predict, lstm_units, number_of_layers,
               str(dropout_rate).replace('.', ''), str(learning_rate).replace('.', ''),
               'percent' if percentage_normalizer else '')
    if os.path.exists(output_file_name):
        print("File with name {} already exists!".format(output_file_name))
        sys.exit(1)

    return output_file_name