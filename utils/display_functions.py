import matplotlib.pyplot as plt


def print_shape_describe_head(df, title=""):
    print("\n{}\n{}\n{}\n{}\n".format(title, df.shape, df.describe(), df.head()))


def print_shape(df, title=""):
    print("\n{}\n{}\n".format(title, df.shape))


def visualize_results(series, labels, colors=None, title='Price prediction', block=True):
    plt.figure(figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')
    if not colors:
        colors = ['red', 'black', 'blue', 'yellow'][:len(series)]
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

    plt.plot(model_fit_history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show(block=block)
