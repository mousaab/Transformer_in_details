import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # d_k is the dimension of each head

        # weight matrices W_q, W_k, and W_v are used to project\
        #  Q, K, and V into d_model-dimensional space
        self.W_q = nn.Linear(d_model, d_model)  # W_q is the linear layer for Q
        self.W_k = nn.Linear(d_model, d_model)  # W_k is the linear layer for K
        self.W_v = nn.Linear(d_model, d_model)  # W_v is the linear layer for V
        self.W_o = nn.Linear(d_model, d_model)  # W_o is the linear layer for output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """_summary_

        The -2 and -1 indices indicate that we want to swap the last two dimensions of the tensor K.
        This effectively changes the shape of K from (batch_size, num_heads, seq_length, d_k) to
        (batch_size, num_heads, d_k, seq_length).
        This shape is suitable for matrix multiplication with Q,  as it aligns the dimensions
        for the dot product.
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # These scores represent the strength of\
        # attention from each query to each key.
        if mask is not None:
            # """_summary_
            # If a mask is provided, this line masks certain values in the attention scores.\
            # The purpose of this masking is to exclude positions in the input sequence \
            # that should not receive attention. Typically, a mask has zeros at positions \
            # that should be masked and ones elsewhere. By setting the masked values to\
            # a very negative number (-1e9 in this case), it effectively makes \
            # them have negligible influence on the softmax operation.
            # """
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)  # It uses the softmax function to convert \
        # the attention scores into a probability distribution.
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # The purpose of this code is to prepare the query, key, and value tensors for \
        # multi-head attention. In multi-head attention, the input tensors are linearly \
        # transformed separately for each attention head to capture different aspects of the data. \
        # The splitting of these transformed tensors into heads allows the model to focus on\
        #  different parts of the input sequence simultaneously, which is a key feature of the Transformer architecture.

        # This code snippet applies linear transformations to the input tensors Q, K, and V \
        # and then splits the transformed tensors into multiple heads to facilitate multi-head \
        # attention computations later in the forward pass of the neural network.

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
