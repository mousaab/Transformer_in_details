import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self, d_model, d_ff
    ):  # d_model is the dimension of the input and output vectors(layers), and d_ff is the
        # dimension of the intermediate vector hidden layer in the feedforward network.
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(
        self, x
    ):  # input tensor x, which represents the output of the self-attention mechanism for a sequence of tokens.
        return self.fc2(self.relu(self.fc1(x)))
