import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


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
        self.loss_fct = CrossEntropyLoss(reduction='sum')

    def forward(self, input_ids, labels=None):
        embedded_inp = self.drop(self.embedding(input_ids))
        output_hidden_states = self.rnn(embedded_inp)[0]
        lm_logits = self.decoder(output_hidden_states)

        loss = None
        multiview_loss = None
        if self.training and labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens

            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            num_terms = torch.sum(shift_labels != -100)
            if num_terms:
                loss = loss / num_terms
            else:
                loss = 0.0

        # Match the format of GPT output
        return loss, multiview_loss, lm_logits
