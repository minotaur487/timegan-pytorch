from torch import nn
from utils import init_hidden, select_rnn_layer


class Embedder(nn.Module):
    """
    Embedding Network: Feature space to latent space.

    Args:
      - X: input time-series features; tensor
      - parameters:
        - input_size: dimension of input layer; integer
        - rnn_type: key that indicates RNN type; constant string
        - hidden_size: dimension of hidden layers; integer
        - sequence_length: tensor corresponding to each sequence in X
            where the value at the corresponding index is the length of
            the sequence; torch.tensor
        - n_layers: number of recurrent layers; integer

    Returns:
      - H: latent representation logits; tensor
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

    def forward(self, X, parameters):
        sequence_lengths = parameters.sequence_length
        device = X.device

        batch_size = X.size(0)
        rnn_init_hidden = init_hidden(device, batch_size, parameters)

        packed_input = nn.rnn_utils.pack_padded_sequence(
            X, sequence_lengths, batch_first=True, enforce_sorted=False
        )
        H, _ = self.rnn_stack(packed_input, rnn_init_hidden)
        return H
