import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorForLanguageModeling:

    def __init__(self, tokenizer, rap_no_grad=True, model_type='transformer'):
        self.tokenizer = tokenizer
        self.rap_no_grad = rap_no_grad
        self.model_type = model_type

    def __call__(self, examples):
        batch = self._tensorize_batch([example['input_ids'] for example in examples])
        labels = batch.clone().detach()
        # Remove pad tokens and start tokens from loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == self.tokenizer.bos_token_id] = -100

        if self.rap_no_grad and 'piece_type_posns' in examples[0]:
            for idx, example in enumerate(examples):
                labels[idx, example['piece_type_posns']] = -100

        output_dict = {"input_ids": batch, "labels": labels}
        return output_dict

    def _tensorize_batch(self, examples):
        padded_sequence = pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if self.model_type != 'reformer':
            return padded_sequence
        else:
            max_len = padded_sequence.shape[1]
            # increased_len = 350 - max_len
            # additional_padding = torch.Tensor(padded_sequence.shape[0], increased_len).fill_(
            #     self.tokenizer.pad_token_id)
            # return torch.cat([padded_sequence, additional_padding.long()], dim=1)
            if max_len % 50 == 0:
                return padded_sequence
            else:
                increased_len = (max_len//50 + 1) * 50 - max_len
                additional_padding = torch.Tensor(padded_sequence.shape[0], increased_len).fill_(
                    self.tokenizer.pad_token_id)
                return torch.cat([padded_sequence, additional_padding.long()], dim=1)
