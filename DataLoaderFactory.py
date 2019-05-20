from data_loader import BitcoinDataLoader
from ripple_data_loader import RippleDataLoader


def get_data_loader(csv_file_path, days_to_predict, sequence_length=60):
    if "bitcoin" in csv_file_path:
        return BitcoinDataLoader(csv_file_path, days_to_predict, sequence_length)
    elif 'ripple' in csv_file_path:
        return RippleDataLoader(csv_file_path, days_to_predict)
