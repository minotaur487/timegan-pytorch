from torch import nn
from utils import init_hidden, select_rnn_layer


class Recovery(nn.Module):
    """
    Recovery Network: Latent space to feature space.

    Args:
      - H: latent representation; tensor
      - parameters:
        - input_size: dimension of input layer; integer
        - rnn_type: key that indicates RNN type; constant string
        - hidden_size: dimension of hidden layers; integer
        - sequence_length: tensor corresponding to each sequence in X
            where the value at the corresponding index is the length of
            the sequence; torch.tensor
        - n_layers: number of recurrent layers; integer

    Returns:
      - X_tilde: recovered data; tensor
    """

    def __init__(self, parameters):
        super(self).__init__()
        selected_rnn = select_rnn_layer(
            rnn_type=parameters["rnn_type"],
            input_size=parameters["hidden_size"],
            hidden_size=parameters["hidden_size"],
            n_layers=parameters["n_layers"],
            batch_first=True,
        )
        self.rnn_stack = nn.Sequential(
            selected_rnn,
            nn.Linear(parameters["hidden_size"], parameters["input_size"]),
            nn.sigmoid(),
        )

    def forward(self, H, parameters):
        sequence_lengths = parameters.sequence_length
        device = H.device

        batch_size = H.size(0)
        rnn_init_hidden = init_hidden(device, batch_size, parameters)

        packed_input = nn.rnn_utils.pack_padded_sequence(
            H, sequence_lengths, batch_first=True, enforce_sorted=False
        )
        X_tilde, _ = self.rnn_stack(packed_input, rnn_init_hidden)
        return X_tilde
