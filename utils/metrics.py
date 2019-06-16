from sklearn.metrics import confusion_matrix, f1_score
import numpy as np


def get_mse_series(actual, predicted):
    return np.array([(y1 - y2) ** 2 for y1, y2 in zip(actual, predicted)])


def get_confusion_matrix_f1score_for_percentage_prediction(actual_percentage, predicted_percentages):
    binary_actual = []
    binary_predicted = []
    for idx in range(len(actual_percentage)):
        if actual_percentage[idx] < 0:
            binary_actual.append(1)
        else:
            binary_actual.append(0)

        if predicted_percentages[idx] < 0:
            binary_predicted.append(1)
        else:
            binary_predicted.append(0)

    return confusion_matrix(binary_actual, binary_predicted).ravel(), f1_score(binary_actual, binary_predicted)


def get_binary_accuracy_from_percentage_prediction(actual_percentages, predicted_percentages):
    correct = 0
    for actual, predicted in zip(actual_percentages, predicted_percentages):
        actual = actual[0]
        predicted = predicted[0]
        if (actual > 0 and predicted > 0) or (actual < 0 and predicted < 0):
            correct += 1
            print('ok', actual, predicted)
        else:
            print('nok', actual, predicted)
    return correct / len(actual_percentages)


def get_binary_accuracy_from_price_prediction(input_prices, actual_price, predicted_price):
    correct = 0
    for idx in range(len(input_prices)):
        if actual_price[idx] < input_prices[idx][-1] and predicted_price[idx] < input_prices[idx][-1] or \
                actual_price[idx] > input_prices[idx][-1] and predicted_price[idx] > input_prices[idx][-1]:
            correct += 1
    return correct / len(input_prices)


def get_confusion_matrix_f1score_for_price_prediction(input_prices, actual_price, predicted_price):
    binary_actual = []
    binary_predicted = []
    for idx in range(len(input_prices)):
        if actual_price[idx] < input_prices[idx][-1]:
            binary_actual.append(1)
        else:
            binary_actual.append(0)
        if predicted_price[idx] < input_prices[idx][-1]:
            binary_predicted.append(1)
        else:
            binary_predicted.append(0)
    return confusion_matrix(binary_actual, binary_predicted).ravel(), f1_score(binary_actual, binary_predicted)
