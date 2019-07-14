import random

import matplotlib.pyplot as plt


def print_shape_describe_head(df, title=""):
    print("\n{}\n{}\n{}\n{}\n".format(title, df.shape, df.describe(), df.head()))


def print_shape(df, title=""):
    print("\n{}\n{}\n".format(title, df.shape))
    print("[0]: {}\n[1]: {}\n[-1]: {}".format(df[0], df[1], df[-1]))


def plot_result_lines(actual, predicted, block=False):
    plt.figure()
    for y1, y2, x in zip(actual, predicted, range(len(actual))):
        plt.vlines(x, ymin=y1, ymax=y2, colors='red')
    plt.plot(actual, color='black', label='actual price')
    plt.plot(predicted, color='red', label='predicted price')
    plt.legend(loc=0, prop={'size': 20})
    plt.title('Actual-Predicted diff', fontsize=30)
    plt.show(block=block)


def visualize_results(series, labels, colors=None, title='Price prediction', block=True):
    plt.figure(figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')
    if not colors:
        colors = ['black', 'red', 'blue', 'yellow'][:len(series)]
    for one_series, label, color in zip(series, labels, colors):
        plt.plot(one_series, color=color, label=label)
        plt.plot(one_series, '+', color=color, label=label)

    plt.title(title, fontsize=40)

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
    plt.xlabel('Time', fontsize=40)
    plt.ylabel('BTC Price(USD)', fontsize=40)
    plt.legend(loc=2, prop={'size': 25})
    plt.show(block=block)


def display_model_train_history_loss(model_fit_history, block=True):
    print("Displaying model train loss history")

    plt.plot(model_fit_history.history['val_loss'], label='validation loss')
    plt.plot(model_fit_history.history['loss'], label='training loss')
    plt.legend(loc=1, prop={'size': 25})
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show(block=block)


def display_model_train_history_acc(model_fit_history, block=True):
    print("Displaying model train acc history")

    plt.plot(model_fit_history.history['val_acc'], label='validation acc')
    plt.plot(model_fit_history.history['acc'], label='training acc')
    plt.legend(loc=1, prop={'size': 25})
    plt.title('Model acc')
    plt.ylabel('acc')
    plt.xlabel('Epoch')
    plt.show(block=block)


def display_plots_with_single_prediction(input_prices, actual_price, predicted_price):
    total_plots = min(3, len(input_prices)//9)
    indexes = list(range(len(input_prices)))
    random.shuffle(indexes)

    rand_idx = 0
    for cur_plot in range(total_plots):
        f, axarr = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                idx = indexes[rand_idx]
                axarr[i, j].plot(input_prices[idx], label='previous price used for prediction')
                axarr[i, j].plot(len(input_prices[idx]), actual_price[idx], 'P', color='green', label='actual')
                axarr[i, j].plot(len(input_prices[idx]), predicted_price[idx], 'x', color='red', label='predicted')
                axarr[i, j].set_title("Prediction {}".format(idx))
                rand_idx += 1
        plt.show(block=True)




    # for i in range(len(input_prices)):
        # print("actual vs predicted for {}: {} vs {}".format(input_prices[i], actual_price[i], predicted_price[i]))

    #
    # previous_price = previous_price[:-predicted_days*2]
    # plots = list()
    # for i in range(predicted_days):
    #     p1 = previous_price[:-predicted_days]
    #
    # f, axarr = plt.subplots(2, 2)
    # axarr[0, 0].plot(x, y)
    # axarr[0, 0].set_title('Axis [0,0]')
    # axarr[0, 1].scatter(x, y)
    # axarr[0, 1].set_title('Axis [0,1]')
    # axarr[1, 0].plot(x, y ** 2)
    # axarr[1, 0].set_title('Axis [1,0]')
    # axarr[1, 1].scatter(x, y ** 2)
    # axarr[1, 1].set_title('Axis [1,1]')


def display_confusion_matrix(conf_matrix):
    tn, fp, fn, tp = conf_matrix
    print('\npositive = price will go down')
    print("Total samples: {}".format(tn + tp + fp + fn))
    print("True negatives: {}".format(tn))
    print("True positives: {}".format(tp))
    print("False positives: {}".format(fp))
    print("False negatives: {}".format(fn))
