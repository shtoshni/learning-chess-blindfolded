import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, rnn_type, vocab_size, n_embd, n_hid, n_layer, rnn_dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(rnn_dropout)
        self.embedding = nn.Embedding(vocab_size, n_embd)
        if rnn_type in ['lstm', 'gru']:
            self.rnn = getattr(nn, rnn_type.upper())(
                input_size=n_embd, hidden_size=n_hid, num_layers=n_layer, dropout=rnn_dropout,
                batch_first=True
            )
        else:
            raise ValueError

        # Weights are not tied
        self.decoder = nn.Linear(n_hid, vocab_size)

    def forward(self, input_ids, **kwargs):
        embedded_inp = self.drop(self.embedding(input_ids))
        output_hidden_states = self.rnn(embedded_inp)[0]
        lm_logits = self.decoder(output_hidden_states)

        return lm_logits
