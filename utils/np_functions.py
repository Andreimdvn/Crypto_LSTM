import numpy as np


def get_price_series_from_start_price_and_percentage(start_price, percentage_array):
    price_series = []
    # print('start_price', start_price)
    for percent in percentage_array:
        percent = float(percent)
        price_series.append(start_price + start_price * percent)
        start_price = price_series[-1]
    # print('percentage array', percentage_array)
    # print('price series', price_series)

    return np.array(price_series)


def get_price_series_from_prices_and_percentages(prices, percentage_array):
    price_series = []
    print("get price series:", len(prices))
    print('percentage array', len(percentage_array))
    for price, percent_change in zip(prices, percentage_array):
        price_series.append(price + percent_change * price)

    return price_series
