import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, rnn_type, vocab_size, n_embd, n_layers, rnn_dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(rnn_dropout)
        self.encoder = nn.Embedding(vocab_size, n_embd)
        if rnn_type in ['lstm', 'gru']:
            self.rnn = getattr(nn, rnn_type.upper())(
                input_size=n_embd, hidden_size=n_embd, num_layers=n_layers, dropout=rnn_dropout)
        else:
            raise ValueError

        self.decoder = nn.Linear(n_embd, vocab_size)
        # Tie weights
        self.decoder.weight = self.encoder.weight

        self.rnn_lm_model = nn.Sequential(
            self.encoder,
            self.drop,
            self.rnn,
            self.drop,
            self.decoder
        )

    def forward(self, input_ids, hidden):
        decoded = self.rnn_lm_model(input_ids)
        decoded = decoded.view(-1, self.ntoken)
        return torch.nn.functional.log_softmax(decoded, dim=1), hidden

