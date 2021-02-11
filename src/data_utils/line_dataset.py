import logging
import os
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from os import path
import numpy as np

logger = logging.getLogger(__name__)


class LineByLineTextDataset(Dataset):

    def __init__(self, tokenizer, file_path, block_size, max_instances=None,
                 rap_prob=0.0):
        # print(file_path)
        assert os.path.isfile(file_path)
        self.rap_prob = rap_prob
        self.tokenizer = tokenizer

        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            if max_instances:
                self.lines = self.lines[:max_instances]

        batch_encoding = tokenizer(self.lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.end_positions = batch_encoding["end_positions"]

        if self.rap_prob:
            rap_dir = path.join(path.dirname(path.dirname(file_path)), "rap")
            rap_file = path.join(rap_dir, path.splitext(path.basename(file_path))[0] + ".npy")
            self.rap_data = np.load(rap_file, allow_pickle=True)[:max_instances]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        output_dict = {}

        if self.rap_prob:
            self.examples[i], self.end_positions[i], piece_type_posns = self._process_rap(
                self.rap_data[i], self.examples[i], self.end_positions[i])
            output_dict['piece_type_posns'] = piece_type_posns

        output_dict['input_ids'] = torch.tensor(self.examples[i])
        output_dict['separator_ind'] = self.end_positions[i]

        return output_dict

    def _process_rap(self, rap_list, example, end_positions):
        use_piece_type_list = np.random.choice([0, 1], size=(len(rap_list),), p=[1 - self.rap_prob, self.rap_prob])
        piece_type_list = [self.tokenizer.vocab[piece_type] if (piece_type is not None and use_piece_type) else -1
                           for (piece_type, use_piece_type) in zip(rap_list, use_piece_type_list)]

        assert (len(piece_type_list) == sum(end_positions) - 1)

        mod_example = list(example)
        mod_end_positions = list(end_positions)

        offset = 1
        move_counter = 0
        piece_type_posns = []
        for idx, end_position in enumerate(end_positions):
            # Check if it's end position
            if (end_position == 1) and (move_counter < len(piece_type_list)):
                piece_type = piece_type_list[move_counter]
                if piece_type != -1:
                    # Insert piece type
                    mod_example = mod_example[:idx + offset] + [piece_type] + mod_example[idx + offset:]
                    mod_end_positions = mod_end_positions[:idx + offset] + [0] + mod_end_positions[idx + offset:]
                    piece_type_posns.append(idx + offset)

                    offset += 1

                move_counter += 1

        return mod_example, mod_end_positions, piece_type_posns

    def get_last_mention_idx(self, example):
        tokenizer = self.tokenizer
        tokens = [tokenizer.id2symbol[idx] for idx in example]
        is_posn_list = [1 if len(token) == 2 else 0 for token in tokens]
        is_move_end_list = [0] * len(is_posn_list)
        counter = 0
        for idx, posn_token in enumerate(is_posn_list):
            if posn_token:
                counter += 1
                if counter % 2 == 0:
                    is_move_end_list[idx] = 1

        last_mention_dict = {}
        last_mention_list = []
        for idx, (token, is_posn, is_move_end) in enumerate(zip(tokens, is_posn_list, is_move_end_list)):
            last_mention_idx = -1
            if is_posn:
                if is_move_end:
                    last_mention_dict[token] = idx
                else:
                    # Starting token
                    if token in last_mention_dict:
                        last_mention_idx = last_mention_dict[token]

            last_mention_list.append(last_mention_idx)

        return last_mention_list

