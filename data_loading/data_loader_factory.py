from data_loading.local_bitcoin_percentage_data_loader import BitcoinPercentageDataLoader
from data_loading.crypto_data_loader import CryptoDataLoader
from data_loading.local_bitcoin_data_loader import LocalBitcoinDataLoader
from data_loading.local_ripple_data_loader import LocalRippleDataLoader


def get_data_loader(csv_source, days_to_predict, percentage_normalizer, sequence_length):
    if 'datahub' in csv_source:
        return CryptoDataLoader(csv_source, days_to_predict, sequence_length, use_percentage=percentage_normalizer)
    else:
        if percentage_normalizer:
            return BitcoinPercentageDataLoader(csv_source, days_to_predict, sequence_length)
        else:
            if "bitcoin" in csv_source:
                return LocalBitcoinDataLoader(csv_source, days_to_predict, sequence_length)
            elif 'ripple' in csv_source:
                return LocalRippleDataLoader(csv_source, days_to_predict)
