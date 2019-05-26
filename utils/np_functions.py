import numpy as np


def get_price_series_from_start_price_and_percentage(start_price, percentage_array):
    price_series = []
    for percent in percentage_array:
        price_series.append(start_price + start_price * percent * 100)
        start_price = price_series[-1]

    return np.array(price_series)
