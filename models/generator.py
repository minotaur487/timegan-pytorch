from torch import nn
from utils import init_hidden, select_rnn_layer


class Generator(nn.Module):
    """
    Generator Network: Generate time-series data in latent space.

    Args:
      - Z: random variables, same shape as input; tensor
      - parameters:
        - input_size: dimension of input layer; integer
        - rnn_type: key that indicates RNN type; constant string
        - hidden_size: dimension of hidden layers; integer
        - sequence_length: tensor corresponding to each sequence in X
            where the value at the corresponding index is the length of
            the sequence; torch.tensor
        - n_layers: number of recurrent layers; integer

    Returns:
      - E: generated embedding; tensor
    """

    def __init__(self, parameters):
        super(self).__init__()
        selected_rnn = select_rnn_layer(
            rnn_type=parameters["rnn_type"],
            input_size=parameters["input_size"],
            hidden_size=parameters["hidden_size"],
            n_layers=parameters["n_layers"],
            batch_first=True,
        )
        self.rnn_stack = nn.Sequential(
            selected_rnn,
            nn.Linear(parameters["hidden_size"], parameters["hidden_size"]),
            nn.sigmoid(),
        )

    def forward(self, Z, parameters):
        sequence_lengths = parameters.sequence_length
        device = Z.device

        batch_size = Z.size(0)
        rnn_init_hidden = init_hidden(device, batch_size, parameters)

        packed_input = nn.rnn_utils.pack_padded_sequence(
            Z, sequence_lengths, batch_first=True, enforce_sorted=False
        )
        E, _ = self.rnn_stack(packed_input, rnn_init_hidden)
        return E
