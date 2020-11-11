import torch
import glob
import random

from collections import OrderedDict
from os import path
from pytorch_lightning.core.datamodule import LightningDataModule
# from transformers import DataCollatorForLanguageModeling
import numpy as np

from data_utils.data_collator import DataCollatorForLanguageModeling
from data_utils.line_dataset import LineByLineTextDataset
from data_utils.chess_tokenizer import ChessTokenizer
from constants import GAME_TYPES


class ChessLMDataModule(LightningDataModule):
    def __init__(self,  data_dir=None, vocab_dir=None, batch_size=8, num_workers=1,
                 train_size=1e6, notation="uci", n_positions=800, other_eval=False,
                 rem_train_id_path=None, rap_prob=0.0, rap_no_grad=False,
                 multiview=False, oracle=False,
                 fixed_attention=False,
                 **kwargs):
        super().__init__()

        self.vocab_dir = vocab_dir
        self.data_dir = data_dir
        self.train_size = train_size

        # Additional model settings
        self.rap_prob = rap_prob
        self.multiview = multiview
        self.oracle = oracle
        self.grounding = (self.multiview or self.oracle)    # Both use board state
        self.notation = notation

        if self.oracle or self.rap_prob > 0.0:
            self.notation = "rap"

        self.max_len = n_positions
        self.fixed_attention = fixed_attention

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer = ChessTokenizer(self.vocab_dir, notation=self.notation)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, rap_no_grad=rap_no_grad,
            grounding=self.grounding, fixed_attention=fixed_attention,
        )

        # Set the file names up
        self.rem_train_id_path = rem_train_id_path

        self.train_file = path.join(self.data_dir, f"train_medium.txt")
        self.dev_file = path.join(self.data_dir, "dev.txt")
        self.test_file = path.join(self.data_dir, "test.txt")

        self.num_of_canonical_tokens = self.get_num_of_canonical_tokens()

        if other_eval:
            self.other_eval_files = {}
            if self.oracle:
                final_dir = "rap_100"  # Add piece type to all the sequences
            else:
                final_dir = "rap_0" if self.notation == 'rap' else 'uci'

            other_eval_dir = path.join(path.dirname(self.data_dir), f"other_eval/{final_dir}")
            cloze_files = sorted(glob.glob(path.join(other_eval_dir, "dev_*")))

            if GAME_TYPES[0] in self.data_dir:
                train_type, eval_type = GAME_TYPES
            else:
                eval_type, train_type = GAME_TYPES

            self.other_eval_files[train_type] = cloze_files
            self.other_eval_files[eval_type] = []

            for cloze_file in cloze_files:
                self.other_eval_files[eval_type].append(cloze_file.replace(train_type, eval_type))

            self.other_eval_fen = {train_type: {}, eval_type: {}}
            for game_type in [train_type, eval_type]:
                fen_dir_train_type = path.join(path.dirname(self.data_dir), "fen")
                fen_dir_game_type = fen_dir_train_type.replace(train_type, game_type)
                for category in ['easy', 'hard']:
                    self.other_eval_fen[game_type][category] = np.load(
                        path.join(fen_dir_game_type, f'other_eval_{category}.npy'), allow_pickle=True)

            self.other_eval_sets = OrderedDict()
            self.load_other_eval_sets()
            print("Other eval sets loaded!")

    def get_num_of_canonical_tokens(self):
        split_to_num_tokens = {}
        for split in ['val', 'test']:
            data_file = None
            if split == 'val' or split == 'dev':
                data_file = self.dev_file
            elif split == 'test':
                data_file = self.test_file
            else:
                raise ValueError

            num_moves = []
            with open(data_file) as f:
                for line in f:
                    # (TODO): ADDING 1 for EOS - WHETHER TO DO IT OR NOT
                    num_moves.append(len(line.strip().split()) + 1)

            split_to_num_tokens[split] = num_moves

        return split_to_num_tokens

    def train_dataloader(self):
        train_dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer, file_path=self.train_file, block_size=self.max_len,
            max_instances=self.train_size,
            grounding=self.grounding, rap_prob=(1.00 if self.oracle else self.rap_prob),
            fixed_attention=self.fixed_attention)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, collate_fn=self.data_collator, drop_last=False, pin_memory=True)

        return train_loader

    def val_dataloader(self):
        dev_dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer, file_path=self.dev_file, block_size=self.max_len,
            grounding=self.grounding, rap_prob=(1.00 if self.oracle else 0.0),
            fixed_attention=self.fixed_attention)
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, collate_fn=self.data_collator, drop_last=False, pin_memory=True)
        # print("Number of dev batches: %d" %len(dev_loader))

        return dev_loader

    def load_other_eval_sets(self):
        for game_type in GAME_TYPES:
            self.other_eval_sets[game_type] = OrderedDict()
            for eval_file in self.other_eval_files[game_type]:
                category = 'easy' if 'easy' in path.basename(eval_file) else 'hard'
                fen_data = self.other_eval_fen[game_type][category]
                max_len = 0
                eval_set_name = self.get_eval_set_name(eval_file)
                eval_set = []
                ignore_first = True
                fields = None
                with open(eval_file) as f:
                    for idx, line in enumerate(f):
                        if ignore_first:
                            ignore_first = False
                            fields = line.strip().split(",")[1:]
                            continue

                        instance = line.strip().split(",")
                        prefix = instance[0]
                        field_vals = instance[1:]
                        prefix_enc, move_end_positions = self.tokenizer.encode(
                            prefix, add_special_tokens=False, get_move_end_positions=True)
                        prefix_enc = ([self.tokenizer.bos_token_id] + prefix_enc)

                        separator_ind_list = [1] + move_end_positions  # The [1] reflects that <s> is a move ender
                        board_rep = LineByLineTextDataset._process_fen(
                            fen_data[idx], separator_ind_list)

                        # print(separator_ind_list)
                        # pad_board_rep = torch.zeros_like(board_rep[0])

                        board_rep_list = DataCollatorForLanguageModeling.align_board_state(
                            board_rep, separator_ind_list
                        )

                        board_rep_aligned = torch.unsqueeze(torch.stack(board_rep_list, dim=0), dim=0)

                        gt_options_dict = OrderedDict()
                        for (field, field_val) in zip(fields, field_vals):
                            gt_options_str = field_val.split()
                            gt_options_enc = set()
                            for gt_option in gt_options_str:
                                option_enc = self.tokenizer.encode(
                                    gt_option, add_special_tokens=False, get_move_end_positions=False)
                                max_len = max(max_len, len(option_enc))
                                # if len(option_enc) == 4:
                                #     print(gt_option)
                                gt_options_enc.add(tuple(option_enc))

                            gt_options_dict[field] = (gt_options_enc, field_val)

                        eval_set.append([prefix_enc, board_rep_aligned, gt_options_dict, prefix])

                # print(eval_set_name, max_len)
                self.other_eval_sets[game_type][eval_set_name] = eval_set

    def test_dataloader(self):
        test_dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer, file_path=self.test_file, block_size=self.max_len,
            grounding=self.grounding, rap_prob=(1.00 if self.oracle else 0.0),
            fixed_attention=self.fixed_attention)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, collate_fn=self.data_collator, drop_last=False)

        return test_loader

    @staticmethod
    def get_eval_set_name(filename):
        name = path.basename(filename)
        name = name[4:-4]  # Remove dev_ and .txt
        return ' '.join([part.capitalize() for part in name.split('_')])
