import torch
from torch import nn

LSTM_TYPE = "lstm"
LSTMNL_TYPE = "lstmnl"
RNN_TYPE = "rnn"
GRU_TYPE = "gru"


def init_hidden(device, batch_size, rnn_type, n_layers, hidden_size):
    """
    Returns the initial hidden state for our RNN

    Args:
      - device: input data tensor's device
    - hidden_size: dimension of hidden layers; integer
    - n_layers: number of recurrent layers; integer
    - rnn_type: key that indicates RNN type; constant string
    """
    if rnn_type == LSTM_TYPE or rnn_type == LSTMNL_TYPE:
        h0 = torch.zeros(n_layers, batch_size, hidden_size).to(device)
        c0 = torch.zeros(n_layers, batch_size, hidden_size).to(device)
        hidden = (h0, c0)
    else:
        # Otherwise RNN or GRU is selected
        hidden = torch.zeros(n_layers, batch_size, hidden_size).to(device)

    return hidden


def select_rnn_layer(rnn_type, input_size, hidden_size, n_layers):
    """
    Returns the desired configured RNN.

    Args:
    - hidden_size: dimension of hidden layers; integer
    - n_layers: number of recurrent layers; integer
    - rnn_type: key that indicates RNN type; constant string
    - input_size: dimension of input layer; integer
    """
    if rnn_type == RNN_TYPE:
        return nn.RNN(
            input_size, hidden_size, n_layers, nonlinearity="tanh", batch_first=True
        )
    elif rnn_type == GRU_TYPE:
        return nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
    elif rnn_type == LSTM_TYPE:
        return nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
    elif rnn_type == LSTMNL_TYPE:
        return None
    return None
