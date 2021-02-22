import torch
import glob
import random
import json
from collections import OrderedDict
from os import path
from pytorch_lightning.core.datamodule import LightningDataModule
import numpy as np

from data_utils.data_collator import DataCollatorForLanguageModeling
from data_utils.line_dataset import LineByLineTextDataset
from data_utils.chess_tokenizer import ChessTokenizer
from constants import LENGTH_CATEGORIES, TASK_CATEGORIES


class ChessLMDataModule(LightningDataModule):
    def __init__(self,  data_dir=None, vocab_dir=None, batch_size=8, num_workers=1,
                 train_size=1e6, n_positions=800, other_eval=True,
                 rap_prob=0.0, rap_no_grad=False, oracle=False,
                 model_type='transformer',
                 **kwargs):
        super().__init__()

        # Set Other eval
        self.other_eval = other_eval
        self.model_type = model_type

        self.vocab_dir = vocab_dir
        self.data_dir = data_dir
        self.train_size = train_size

        # Additional model settings
        self.rap_prob = rap_prob
        self.oracle = oracle
        if self.oracle:
            self.rap_prob = 1.00

        self.max_len = n_positions

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer = ChessTokenizer(path.join(self.vocab_dir, "vocab.txt"))
        self.train_data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, rap_no_grad=rap_no_grad,
            model_type=self.model_type
        )

        # We don't want loss calculated over piecetypes during inference (rap_no_grad=True) for RAP models.
        # Piecetypes are not considered a part of prediction, rather just the extra information present.
        # In the current setup, piecetypes are a part of inference only for the "Oracle" model.
        self.inference_data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, rap_no_grad=(False if self.oracle else True),
            model_type=self.model_type
        )

        # Set the file names up
        self.train_file = path.join(self.data_dir, "train.txt")
        self.dev_file = path.join(self.data_dir, "dev.txt")
        self.test_file = path.join(self.data_dir, "test.txt")

        self.num_of_canonical_tokens = self.get_num_of_canonical_tokens()

        if self.other_eval:
            self.other_eval_files, self.other_eval_fen = {}, {}
            other_eval_dir = path.join(path.dirname(self.data_dir), f"other_eval/uci")

            for task_category in TASK_CATEGORIES:
                if (not self.rap_prob) and (task_category == 'start'):
                    continue
                self.other_eval_files[task_category] = {}
                for len_category in LENGTH_CATEGORIES:
                    self.other_eval_files[task_category][len_category] = path.join(
                        other_eval_dir, f'{task_category}_{len_category}.jsonl')

            self.other_eval_sets = OrderedDict()
            self.load_other_eval_sets()
            # print(self.other_eval_files)
            print("Other eval sets loaded!")

    def get_num_of_canonical_tokens(self):
        split_to_num_tokens = {}
        for split in ['val', 'test']:
            if split == 'val' or split == 'dev':
                data_file = self.dev_file
            elif split == 'test':
                data_file = self.test_file
            else:
                raise ValueError

            num_moves = []
            with open(data_file) as f:
                for line in f:
                    # (NOTE:) ADDING 1 for EOS i.e. EOS is a move as well
                    num_moves.append(len(line.strip().split()) + 1)

            split_to_num_tokens[split] = num_moves

        return split_to_num_tokens

    def load_other_eval_sets(self):
        self.other_eval_sets = OrderedDict()
        for task_category in self.other_eval_files:
            self.other_eval_sets[task_category] = {}
            for len_category in LENGTH_CATEGORIES:
                eval_file = self.other_eval_files[task_category][len_category]
                eval_set = []

                with open(eval_file) as f:
                    for idx, line in enumerate(f):
                        instance = json.loads(line.strip())
                        coded_instance = OrderedDict()  # Dictionary that contains all the information
                        for key, val in instance.items():
                            if "prefix" in key:
                                if (key == "oracle_prefix" and self.oracle) or (key == "prefix" and not self.oracle):
                                    prefix = val
                                    prefix_enc, move_end_positions = self.tokenizer.encode(
                                        prefix, add_special_tokens=False, get_move_end_positions=True)
                                    prefix_enc = ([self.tokenizer.bos_token_id] + prefix_enc)

                                    coded_instance["prefix"] = prefix
                                    coded_instance["prefix_enc"] = prefix_enc
                            else:
                                if isinstance(val, str):
                                    coded_val = self.tokenizer.encode_token(val)
                                    coded_instance[key] = val
                                    coded_instance[key + "_enc"] = coded_val
                                elif isinstance(val, list):
                                    coded_val = [self.tokenizer.encode_token(token) for token in val]
                                    coded_instance[key] = val
                                    coded_instance[key + "_enc"] = coded_val
                                else:
                                    raise ValueError

                        eval_set.append(coded_instance)

                    self.other_eval_sets[task_category][len_category] = eval_set

    def train_dataloader(self):
        train_dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer, file_path=self.train_file, block_size=self.max_len,
            max_instances=self.train_size,
            rap_prob=(1.00 if self.oracle else self.rap_prob))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, collate_fn=self.train_data_collator, drop_last=False, pin_memory=True)

        return train_loader

    def val_dataloader(self):
        dev_dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer, file_path=self.dev_file, block_size=self.max_len,
            rap_prob=(1.00 if self.oracle else 0.0))
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, collate_fn=self.inference_data_collator, drop_last=False, pin_memory=True)

        return dev_loader

    def test_dataloader(self):
        test_dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer, file_path=self.test_file, block_size=self.max_len,
            rap_prob=(1.00 if self.oracle else 0.0))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, collate_fn=self.inference_data_collator, drop_last=False)

        return test_loader
