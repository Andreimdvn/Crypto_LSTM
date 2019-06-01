import matplotlib.pyplot as plt


def print_shape_describe_head(df, title=""):
    print("\n{}\n{}\n{}\n{}\n".format(title, df.shape, df.describe(), df.head()))


def print_shape(df, title=""):
    print("\n{}\n{}\n".format(title, df.shape))
    print("[0]: {}\n[1]: {}".format(df[0][:5], df[1][:5]))


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


def display_model_train_history(model_fit_history, block=True):
    print("Displaying model train history")

    plt.plot(model_fit_history.history['val_loss'], label='validation loss')
    plt.plot(model_fit_history.history['loss'], label='training loss')
    plt.legend(loc=1, prop={'size': 25})
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show(block=block)


def display_plots_with_single_prediction(input_prices, actual_price, predicted_price):
    total_plots = min(3, len(input_prices)//9)
    idx = 0
    for cur_plot in range(total_plots):
        f, axarr = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                axarr[i, j].plot(input_prices[idx])
                axarr[i, j].plot(len(input_prices[idx]), actual_price[idx], 'P', color='green')
                axarr[i, j].plot(len(input_prices[idx]), predicted_price[idx], 'x', color='red')
                axarr[i, j].set_title("Prediction {}".format(idx))
                idx += 1
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