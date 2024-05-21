import numpy as np
from discriminator import Discriminator
from embedder import Embedder
from generator import Generator
from recovery import Recovery
from supervisor import Supervisor


def TimeGAN(raw_data, parameters):
    """
    TimeGAN: Combines all the networks to generate synthetic data

    Args:
      - raw_data: input time-series data; np array
      - parameters:

    Returns:
      - generated_data: synthetic time-series data
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(raw_data).shape

    def train_embedded_network():
        TODO

    def train_supervised_loss():
        TPDP

    def train_joint():
        TODO

    def train_discriminator():
        TODO

    def normalize_data():
        TODO

    def generate_synthetic_data():
        TODO

    def preprocess_data():
        TODO

    X = preprocess_data(raw_data)
    train_embedded_network()
    train_supervised_loss()
    train_joint()
    train_discriminator()
    generated_data = generate_synthetic_data()
    return generate_synthetic_data
