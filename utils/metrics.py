def get_binary_accuracy_from_price_prediction(input_prices, actual_price, predicted_price):
    correct = 0
    for idx in range(len(input_prices)):
        if actual_price[idx] < input_prices[idx][-1] and predicted_price[idx] < input_prices[idx][-1] or \
                actual_price[idx] > input_prices[idx][-1] and predicted_price[idx] > input_prices[idx][-1]:
            correct += 1
    return correct / len(input_prices)
