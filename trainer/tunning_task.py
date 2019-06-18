import argparse
import random
import time

import numpy as np
import pandas as pd

from utils import defaults

from utils.gcloud_utils import copy_file_to_gcloud

lstm_units = [16, 32, 64, 100]
number_of_layers = [1, 2]
dropout_rate = np.linspace(0, 0.25, 6)
learning_rate = [0.001, 0.005, 0.01]
batch_size = [8, 16, 32]
sequence_length = [10, 15, 20, 30, 40, 60]
decay_rate = [0]
# decay_rate = [0, 0.01, 0.05, 0.005]


def get_parameters(used_params, parameters):
    while 1:
        new_choice = dict()
        for param_name in parameters.keys():
            new_choice[param_name] = random.choice(parameters[param_name])
        if new_choice not in used_params:
            return new_choice


def main(csv_data_file, days_to_predict, epochs, job_dir, iterations, percentage_normalizer, log_return,
         results_output_file, prefix_models, classification_model, multiple_features):
    parameters_choice = dict(lstm_units=lstm_units,
                             number_of_layers=number_of_layers,
                             dropout_rate=dropout_rate,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             sequence_length=sequence_length, decay_rate=decay_rate)

    if classification_model:
        from trainer.task_class import main as train_main
    else:
        from trainer.task import main as train_main

    results = []
    used_params = []
    for nr_iter in range(iterations):
        parameters = get_parameters(used_params, parameters_choice)
        print("Iteration {}. New parameters: {}".format(nr_iter, parameters))
        used_params.append(parameters.copy())
        output_file = '{}_model_{}'.format(prefix_models, nr_iter)
        start_time = time.time()

        model_history = train_main(csv_data_file,
                                   days_to_predict=days_to_predict,
                                   epochs=epochs,
                                   batch_size=parameters['batch_size'],
                                   lstm_units=parameters['lstm_units'],
                                   sequence_length=parameters['sequence_length'],
                                   number_of_layers=parameters['number_of_layers'],
                                   dropout_rate=parameters['dropout_rate'],
                                   learning_rate=parameters['learning_rate'],
                                   percentage_normalizer=percentage_normalizer,
                                   log_return=log_return,
                                   output_file=output_file,
                                   use_early_stop=True,
                                   decay_rate=parameters['decay_rate'],
                                   multiple_features=multiple_features,
                                   job_dir=job_dir)
        end_time = time.time()
        parameters['loss'] = model_history.history['loss'][-1]
        parameters['val_loss'] = model_history.history['val_loss'][-1]
        parameters['epochs'] = len(model_history.history['loss'])
        parameters['training_time_min'] = int((end_time - start_time) / 60)
        parameters['model_file'] = output_file
        if classification_model:
            parameters['acc'] = model_history.history['acc'][-1]
            parameters['val_acc'] = model_history.history['val_acc'][-1]

        print("New parameters output: {}".format(parameters))
        results.append(parameters)
        if nr_iter % 10 == 0:
            df = pd.DataFrame.from_dict(results)
            partial_output = "{}_{}.cfg".format(results_output_file.replace(".", '_'), nr_iter)
            df.to_csv(partial_output)
            if job_dir:
                copy_file_to_gcloud(partial_output, job_dir, partial_output)

    df = pd.DataFrame.from_dict(results)
    print(df)
    df.to_csv(results_output_file)
    if job_dir:
        copy_file_to_gcloud(results_output_file, job_dir, results_output_file)


def init_arg_parser():
    parser = argparse.ArgumentParser(description='Tune hyperparameters for model')
    parser.add_argument('-f', '--file_csv', dest='csv_data_file', help='csv file/url', type=str, required=True)
    parser.add_argument('-d', '--days_to_predict', dest='days_to_predict',
                        help='Days to predict. Training set = last number of days',
                        type=int, default=defaults.DEFAULT_DAYS_TO_PREDICT)
    parser.add_argument('-e', '--epochs', dest='epochs', help='Number of epochs used at training',
                        type=int, default=defaults.DEFAULT_EPOCHS_NUMBER)
    parser.add_argument("-j", '--job-dir', dest='job_dir', help='jobs dir used for gcloud training', required=False,
                        default=None)
    parser.add_argument('-i', '--iterations', dest='iterations', help='number of iterations to run the random search',
                        required=True)
    parser.add_argument('-p', '--percentage_prediction', dest='percentage',
                        help='Will convert prices to percentage change', default=False, action='store_true')
    parser.add_argument('-P', '--prefix_models', dest='prefix_models',
                        help='prefix for model name = tuning instance name', default="", type=str)
    parser.add_argument('-L', '--log_return', dest='log_return', help='Will convert prices to log return',
                        default=False, action='store_true')
    parser.add_argument('-o', '--output_file', dest='output_file', help='output file for results of the random search')
    parser.add_argument('-c', '--classification_model', dest='classification_model',
                        help='Use classification model', default=False, action='store_true')
    parser.add_argument('-m', '--multiple_features', dest='multiple_features',
                        help='Use multiple features alongside price', default=False, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = init_arg_parser()
    print(args.__dict__)
    main(args.csv_data_file, int(args.days_to_predict), int(args.epochs), args.job_dir, int(args.iterations),
         args.percentage, args.log_return, args.output_file, args.prefix_models, args.multiple_features,
         args.classification_model)
