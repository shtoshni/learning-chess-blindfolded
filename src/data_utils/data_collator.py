import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorForLanguageModeling:

    def __init__(self, tokenizer, rap_no_grad=True, grounding=False):
        self.tokenizer = tokenizer
        self.rap_no_grad = rap_no_grad
        self.grounding = grounding

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

        if self.grounding:
            max_length = batch.size(1)
            board_rep_list = [example['board_rep'] for example in examples]
            separator_ind_list = [example['separator_ind'] for example in examples]
            board_rep = self._tensorize_board_rep(board_rep_list, separator_ind_list, max_length)
            output_dict["board_rep"] = board_rep

        return output_dict

    def _tensorize_batch(self, examples):
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def _tensorize_board_rep(self, board_rep_list, separator_bool_list, max_length):
        pad_board_rep = torch.zeros_like(board_rep_list[0][0])

        batch_board_rep_list = []
        for board_rep_tens, separator_bool_seq in zip(board_rep_list, separator_bool_list):
            board_rep_list = self.align_board_state(board_rep_tens, separator_bool_seq)

            board_rep_list += [pad_board_rep] * (max_length - len(separator_bool_seq))

            batch_board_rep_list.append(torch.stack(board_rep_list, dim=0))

        return torch.stack(batch_board_rep_list, dim=0)

    @staticmethod
    def align_board_state(board_rep_tens, separator_bool_seq):
        pad_board_rep = torch.zeros_like(board_rep_tens[0])
        board_rep_list = []
        cur_rep = pad_board_rep
        counter = 0
        for separator_bool in separator_bool_seq:
            if separator_bool:
                cur_rep = board_rep_tens[counter, :]
                board_rep_list.append(cur_rep)
                counter += 1
            else:
                board_rep_list.append(pad_board_rep)

        assert(board_rep_tens.shape[0] == sum(separator_bool_seq))
        return board_rep_list
