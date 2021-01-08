import itertools

import torch
import os
import json

from os import path
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict

from pytorch_lightning.core.lightning import LightningModule
from transformers import GPT2Config, get_linear_schedule_with_warmup

from model_utils.gpt2_model import GPT2LMHeadModel
from model_utils.rnn_model import RNNModel
from model_utils.utils import get_strided_attn_mask, get_last_window_attn_mask
from data_utils.simple_datamodule import ChessLMDataModule
from constants import LENGTH_CATEGORIES, TASK_CATEGORIES, MOVE_TYPES

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ChessLM(LightningModule):

    def __init__(self, args,
                 n_embd=768, n_positions=1024, n_layer=12, n_head=12,
                 init_lr=3e-4, accumulate_grad_batches=1,
                 num_training_steps=None, other_eval=True,
                 stride_size=None, window_size=None, multiview_margin=0.5,
                 model_type='transformer',
                 **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.model_type = model_type  # RNN vs Transformer

        # Board state setting
        self.multiview = args.multiview
        self.multiview_loss_wt = args.multiview_loss_wt
        self.oracle = args.oracle
        self.grounding = self.oracle or self.multiview

        self.datamodule = ChessLMDataModule(grounding=self.grounding, **vars(args))
        self.tokenizer = self.datamodule.tokenizer
        vocab_size = len(self.tokenizer.get_vocab())

        self.init_lr = init_lr
        self.accumulate_grad_batches = accumulate_grad_batches
        self.num_training_steps = num_training_steps
        self.other_eval = other_eval
        self.other_eval_log_dir = None

        self.current_epoch_steps = 0

        if self.model_type == 'transformer':
            self.config = GPT2Config(
                vocab_size=vocab_size,
                n_embd=n_embd,
                n_positions=n_positions,
                n_ctx=n_positions,
                n_head=n_head,
                n_layer=n_layer,
            )
            if stride_size is not None:
                attention_mask = get_strided_attn_mask(stride_size, max_seq_length=n_positions)
                self.config.static_attention_mask = attention_mask
            elif window_size is not None:
                attention_mask = get_last_window_attn_mask(window_size, max_seq_length=n_positions)
                self.config.static_attention_mask = attention_mask
            else:
                self.config.static_attention_mask = None

            # Whether to use board state or not
            # Only training
            self.config.multiview = self.multiview
            self.config.multiview_margin = multiview_margin
            self.config.neg_samples = args.neg_samples

            # Both inference and training
            self.config.oracle = args.oracle
            self.config.inject_state = args.inject_state

            # Convolution stuff
            self.config.kernel_size = args.kernel_size
            self.config.out_channels = args.out_channels

            self.max_len = n_positions
            self.model = GPT2LMHeadModel(config=self.config)

        elif model_type == 'rnn':
            self.model = RNNModel(rnn_type=args.rnn_type, vocab_size=vocab_size, n_embd=args.n_embd,
                                  n_hid=args.n_hid, n_layer=n_layer, rnn_dropout=args.rnn_dropout)
        else:
            raise NotImplementedError(f'Model type: {model_type} not supported')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--model_type', type=str, default='transformer', choices= ['transformer', 'rnn'])
        # RNN arguments
        parser.add_argument('--rnn_type', type=str, default='lstm', choices=['gru', 'lstm'])
        parser.add_argument('--rnn_dropout', type=float, default=0.2)
        parser.add_argument('--n_hid', type=int, default=768)

        # Common Transformer and RNN args
        parser.add_argument('--n_layer', type=int, default=12)
        parser.add_argument('--n_embd', type=int, default=768)

        # Transformer arguments
        parser.add_argument('--n_head', type=int, default=12)
        parser.add_argument('--n_positions', type=int, default=512)
        parser.add_argument('--stride_size', type=int, default=None)
        parser.add_argument('--window_size', type=int, default=None)
        # Adding board state
        # (0) RAP
        parser.add_argument('--rap_prob', type=float, default=0.0)
        parser.add_argument('--rap_grad', dest='rap_no_grad', default=True, action="store_false")
        # (1) Multiview
        parser.add_argument('--multiview', action="store_true", default=False)
        parser.add_argument('--multiview_margin', type=float, default=0.5)
        parser.add_argument('--multiview_loss_wt', type=float, default=1.0)
        parser.add_argument('--neg_samples', type=int, default=10)
        # (2) Oracle
        parser.add_argument('--oracle', default=False, action="store_true")
        parser.add_argument('--inject_state', type=str, default="just_base",
                            choices=['just_base', 'all'], help="Which layers to add the board state to")
        # Boars state model details
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument('--out_channels', type=int, default=384)

        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.init_lr)
        linear_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1 * self.num_training_steps,
            num_training_steps=self.num_training_steps)
        scheduler = {'scheduler': linear_scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]

    def forward(self, batch):
        outputs = self.model(**batch)
        if self.training:
            return outputs[:2]
        else:
            # Mask out piece type predictions
            # Since we test the model on pure UCI notation, we don't want to penalize RAP-trained/Oracle models
            # for learning to predict the piece types as part of the sequence. So we mask out those indices
            # from the logits.
            labels = batch['labels']
            lm_logits = outputs[2]

            piece_type_mask = torch.tensor(self.tokenizer.token_is_piece_type_mask,
                                           dtype=lm_logits.dtype, device=batch['input_ids'].device)
            lm_logits = lm_logits * (1 - piece_type_mask) + piece_type_mask * (-1e10)

            loss, multiview_loss = outputs[:2]
            if batch['labels'] is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                num_terms = torch.sum(shift_labels != -100)
                if num_terms:
                    loss = loss / torch.sum(shift_labels != -100)
                else:
                    loss = 0.0
            return loss, multiview_loss

    def training_step(self, batch, batch_ids):
        # print(batch_ids)
        if not self.multiview:
            loss, _ = self(batch)
            multiview_loss = None
        else:
            loss, multiview_loss = self(batch)

        train_loss = loss
        train_log = {'loss/train_loss': loss}
        if multiview_loss is not None:
            train_loss = loss + self.multiview_loss_wt * multiview_loss
            train_log['loss/train_multiview_loss'] = multiview_loss

        self.current_epoch_steps += batch["input_ids"].shape[0]
        return {'loss': train_loss, 'log': train_log}

    def validation_step(self, batch, batch_ids, split="val"):
        # print("Validation", batch_ids)
        input_ids, labels = batch["input_ids"], batch["labels"]
        loss, _ = self(batch)
        # Removing labels for which losses are not calculated.
        batch_tokens = torch.sum(labels != -100)
        val_log = {f'loss/{split}_loss': loss.detach()}
        return {f'{split}_loss': loss.detach(), 'log': val_log, 'batch_tokens': batch_tokens,
                'batch_size': input_ids.shape[0]}

    def test_step(self, batch, batch_ids):
        return self.validation_step(batch, batch_ids, split='test')

    def evaluate_on_other_eval_sets(self, to_log=True):
        self.other_eval_log_dir = path.join(self.logger.log_dir, "other_eval")
        if not path.exists(self.other_eval_log_dir):
            os.makedirs(self.other_eval_log_dir)

        other_eval_sets = self.datamodule.other_eval_sets
        performance_dict = OrderedDict()
        for task_category in other_eval_sets:
            performance_dict[task_category] = OrderedDict()

        for (task_category, len_category) in itertools.product(TASK_CATEGORIES, LENGTH_CATEGORIES):
            if task_category not in other_eval_sets:
                # Tasks like starting position for piece type are only defined for models with non-zero RAP
                continue
            eval_set = other_eval_sets[task_category][len_category]
            # Log file
            log_file = path.join(self.other_eval_log_dir, f'{task_category}_{len_category}.jsonl')
            # Prefix modification and Evaluation entries
            prefix_mod_and_eval = {}
            if task_category == "end":
                if self.oracle:
                    # Oracle model also has access to piece type for current move
                    prefix_mod_and_eval[("actual", "end")] = {
                        "add_prefix": ["piece_type", "starting_square"],
                        "eval": {"ExM": "ending_square", "LgM": "legal_ending_square"}
                    }
                    prefix_mod_and_eval[("other", "end")] = {
                        "add_prefix": ["other_piece_type", "other_starting_square"],
                        "eval": {"LgM": "other_ending_square"}
                    }
                else:
                    prefix_mod_and_eval[("actual", "end")] = {
                        "add_prefix": ["starting_square"],
                        "eval": {"ExM": "ending_square", "LgM": "legal_ending_square"}
                    }
                    prefix_mod_and_eval[("other", "end")] = {
                        "add_prefix": ["other_starting_square"],
                        "eval": {"LgM": "other_ending_square"}
                    }

            elif task_category == "start":
                prefix_mod_and_eval[("actual", "start")] = {
                    "add_prefix": ["piece_type"],
                    "eval": {"ExM": "starting_square", "LgM": "legal_starting_square"}
                }
                prefix_mod_and_eval[("other", "start")] = {
                    "add_prefix": ["other_piece_type"],
                    "eval": {"LgM": "other_starting_square"}
                }

            with open(log_file, 'w') as f:
                output_dict_list = list(eval_set)
                for move_type in MOVE_TYPES:
                    eval_metrics = defaultdict(int)
                    for idx, encoded_dict in enumerate(eval_set):
                        prefix = list(encoded_dict["prefix_enc"])
                        prefix_mod_and_eval_val = prefix_mod_and_eval[(move_type, task_category)]
                        prefix_addition = prefix_mod_and_eval_val["add_prefix"]
                        ground_truth = {key: encoded_dict[val]
                                        for key, val in prefix_mod_and_eval_val["eval"].items()}

                        # Add prefix additions such as piece type or starting square to query the model
                        prefix.extend([encoded_dict[entry + "_enc"] for entry in prefix_addition])

                        if self.oracle:
                            board_rep = encoded_dict["board_rep"]
                            logits = self.model(input_ids=torch.tensor([prefix]).cuda(),
                                                board_rep=board_rep.cuda())[2]
                        else:
                            logits = self.model(input_ids=torch.tensor([prefix]).cuda())[2]

                        last_token_logit = logits[0, -1, :]
                        # Get top-k predictions where k=number of legal choices
                        sorted_pred = torch.topk(last_token_logit, k=len(ground_truth["LgM"]),
                                                 dim=0, largest=True, sorted=True)[1].tolist()
                        # prediction_idx_list = sorted_pred
                        # prediction_idx = prediction_idx_list[0]
                        prediction_str_list = [self.tokenizer.decode_token(token_idx) for token_idx in sorted_pred]
                        prediction_str = prediction_str_list[0]

                        output_dict_list[idx][move_type] = {}
                        for match_type, ground_truth_val in ground_truth.items():
                            if match_type == "ExM":
                                corr = int(prediction_str == ground_truth_val)
                                output_dict_list[idx][move_type]["ExM"] = {
                                    "pred": prediction_str, "gt": ground_truth_val, "corr": corr}
                                eval_metrics["ExM"] += corr
                            elif match_type == "LgM":
                                corr = int(prediction_str in ground_truth_val)
                                r_precision = len(set(prediction_str_list).intersection(set(ground_truth_val)))
                                r_precision /= len(set(ground_truth_val))
                                output_dict_list[idx][move_type]["LgM"] = {
                                    "pred": prediction_str_list, "gt": ground_truth_val,
                                    "corr": corr, "r_precision": round(r_precision, 2)}

                                eval_metrics["LgM"] += corr
                                eval_metrics["LgM R-Precision"] += r_precision

                    for metric in eval_metrics:
                        eval_metrics[metric] /= len(eval_set)
                        eval_metrics[metric] *= 100.0  # Percentage

                        global_metric_name = f'{len_category.capitalize()} {move_type.capitalize()} {metric}'
                        performance_dict[task_category][global_metric_name] = round(eval_metrics[metric], 2)
                        print(f'{task_category.capitalize()} {global_metric_name}: '
                              f'{round(eval_metrics[metric], 1)}')
                        if to_log:
                            self.logger.experiment.add_scalar(
                                f'{task_category.capitalize()}/{global_metric_name}', eval_metrics[metric],
                                self.current_epoch
                            )

                for output_dict in output_dict_list:
                    # Need to remove some entries which are not serializable
                    cleaned_dict = OrderedDict()
                    for key, val in output_dict.items():
                        if '_enc' in key:
                            continue
                        if 'board_' in key:
                            continue
                        else:
                            cleaned_dict[key] = val

                    f.write(json.dumps(cleaned_dict) + "\n")

        performance_dict_file = path.join(self.other_eval_log_dir, "perf.json")
        with open(performance_dict_file, 'w') as f:
            f.write(json.dumps(performance_dict))
        return performance_dict

    def validation_epoch_end(self, outputs, split='val'):
        performance_dict = {}
        if (self.global_step > 0 or split == 'test') and self.other_eval:
            performance_dict = self.evaluate_on_other_eval_sets()

        # Calculate validation loss
        val_loss = torch.stack([x[f'{split}_loss'] for x in outputs])
        val_tokens = torch.stack([x['batch_tokens'] for x in outputs])
        num_examples = sum([x['batch_size'] for x in outputs])
        canonical_tokens = self.datamodule.num_of_canonical_tokens[split][:num_examples]

        avg_log_p = torch.sum(val_loss * val_tokens)/torch.sum(val_tokens)
        canonical_log_p = torch.sum(val_loss * val_tokens)/sum(canonical_tokens)

        val_log = {f'loss/token_{split}_loss': avg_log_p, f'loss/{split}_loss': canonical_log_p}
        print("\nToken-level log-loss: %.3f, Move-level log loss: %.3f" % (avg_log_p, canonical_log_p))
        return_dict = {'log': val_log, f'token_{split}_loss': avg_log_p, f'{split}_loss': canonical_log_p}

        # Add performance dictionary i.e. other eval performance to return dictionary i.e. losses
        return_dict.update(performance_dict)
        return return_dict

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, split='test')
