import torch
import torch.nn as nn
from torch.nn import LSTM
import torch.nn.functional as F

import config_part1
import config_part2


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout=0.5, **kwargs):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)

        self.lstm = LSTM(embedding_size, hidden_size, **kwargs)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x)).view(x.shape[0], 1, -1)

        output, (hidden, cell) = self.lstm(embedded)

        return output, (hidden, cell)

    def init_hidden(self, device=config_part1.DEVICE) -> torch.tensor:
        return torch.zeros(1, 1, self.hidden_size, device=device)\
            , torch.zeros(1, 1, self.hidden_size, device=device)


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, dropout=0.5, **kwargs):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embedding_size)

        self.lstm = LSTM(hidden_size + embedding_size, hidden_size, **kwargs)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, enc_output, hidden):
        embedded = self.embedding(x).view(1, 1, -1)

        combined_input = torch.cat([embedded, enc_output], dim=2)

        output, (hidden, cell) = self.lstm(combined_input, hidden)

        predictions = self.dropout(self.fc(output)).view(1, -1)

        predictions = F.log_softmax(predictions, dim=1)

        return predictions, (hidden, cell)

    def init_hidden(self, device=config_part1.DEVICE) -> torch.tensor:
        return torch.zeros(1, 1, self.hidden_size, device=device),\
             torch.zeros(1, 1, self.hidden_size, device=device)
    

class AttentionEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout=0.5, **kwargs):
        super(AttentionEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)

        self.lstm = LSTM(embedding_size, hidden_size, **kwargs)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_length = x.shape[0]
        embedded = self.dropout(self.embedding(x).view(input_length, config_part2.BATCH_SIZE, -1))

        output, (hidden, cell) = self.lstm(embedded)

        return output, (hidden, cell)


class AttentionDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, dropout=0.5, **kwargs):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embedding_size)

        self.lstm = nn.LSTM(hidden_size + embedding_size, hidden_size, **kwargs)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        self.v = nn.Parameter(torch.rand(1, 1, hidden_size))

        self.attention_scores_fc = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.attention_context_fc = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, enc_output, dec_hidden, last_context):
        # get embedding of input
        embedded = self.dropout(self.embedding(x).view(1, config_part2.BATCH_SIZE, -1))

        # create the input to the decoder RNN
        rnn_input = torch.cat([last_context, embedded], dim=-1)

        # get the current decoder hidden state (h_t)
        h_t, dec_hidden = self.lstm(rnn_input, dec_hidden)

        # inflate h_t to be able to concat to the encoder states
        sequence_length = enc_output.shape[0]
        h_repeated = h_t.repeat(sequence_length, 1, 1)

        # get the attention scores
        attention_weights = self.tanh(self.attention_scores_fc(torch.cat([h_repeated, enc_output], dim=-1)))
        attention_weights = attention_weights.permute(1, 2, 0)
        attention_weights = self.v.bmm(attention_weights)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # make the context vector by applying the attention weights to encoder outputs
        enc_output = enc_output.permute(1, 0, 2)
        context = self.tanh(torch.bmm(attention_weights, enc_output))
        context = self.tanh(self.attention_context_fc(context))

        # get the probabilities for the next word in the sequence
        output = F.log_softmax(self.fc(context.view(1, -1)), dim=-1)

        return output.squeeze(), dec_hidden, context, attention_weights.squeeze(1)

    def init_hidden(self, device=config_part2.DEVICE) -> torch.tensor:
        return torch.zeros(1, 1, self.hidden_size, device=device), \
               torch.zeros(1, 1, self.hidden_size, device=device)
