import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, file_name, seq_length=None, preprocess=None):
        """
        Initializes a TimeSeriesDataset object.

        Args:
            file_name (str): The path to the CSV file containing the time series data.
            seq_length (int, optional): The length of each sequence. Defaults to None.
            preprocess (function, optional): A function to preprocess the data. Defaults to None.

        Returns:
            None

        Initializes the TimeSeriesDataset object with the data from the given CSV file.
        If `seq_length` is provided, the data is one sequence and should be preprocessed by
            generating overlapping uniform segments.
        If `seq_length` is not provided, the data should be represented as an array of sequences already.
        If `preprocess` is provided, it is applied to the data before preprocessing.
        """
        self.data = pd.read_csv(file_name)
        self.num_of_sequences = len(self.data)
        self.seq_length = seq_length
        if preprocess:
            self.data = preprocess(self.data, self.seq_length)
            self.num_of_sequences = len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        return sample

    def __len__(self):
        return self.num_of_sequences


def min_max_normalizer(data):
    """
    Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


def timeGAN_preprocess_data(data, seq_length=None):
    """
    Preprocesses the given data for TimeGAN model.

    Args:
        data (numpy.ndarray): The raw input data to be preprocessed.
        seq_length (int, optional): The length of each sequence. Defaults to None.

    Returns:
        numpy.ndarray: The preprocessed data. If `seq_length` is provided, the data is split into overlapping uniform segments.

    This function first normalizes the input data using the `min_max_normalizer` function. Then, if `seq_length` is provided,
    the data is split into overlapping uniform segments using the `generate_overlapping_uniform_segments` function.
    """
    data = min_max_normalizer(data)
    if seq_length:
        data = generate_overlapping_uniform_segments(data, seq_length)
    return data


def generate_overlapping_uniform_segments(sequence, seq_length):
    """
    Generates overlapping uniform segments from a given sequence.

    Args:
        sequence (Sequence): The input sequence from which to generate segments.
        seq_length (int): The length of each segment.

    Returns:
        List[Sequence]: A list of overlapping uniform segments.

    This function takes a sequence and a segment length as input. It iterates over the sequence,
    cutting out segments of the specified length. The segments are overlapping, meaning that
    each segment starts at an index that is less than the length of the sequence minus the segment length.
    The function returns a list of these overlapping uniform segments.
    """
    processed_sequences = []
    for i in range(0, len(sequence) - seq_length):
        segment = sequence[i : i + seq_length]
        processed_sequences.append(segment)
    return processed_sequences
