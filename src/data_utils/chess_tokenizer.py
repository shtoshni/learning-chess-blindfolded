import re
import os
from collections import OrderedDict
from constants import NOTATION_TO_REGEX
from transformers import PreTrainedTokenizer


class ChessTokenizer:
    def __init__(self, vocab_file, notation='uci', pad_token="<pad>", bos_token="<s>",
                 eos_token="</s>", **kwargs):
        # super(ChessTokenizer, self).__init__(
        #     pad_token=pad_token, bos_token=bos_token, eos_token=eos_token, **kwargs)

        self.vocab = OrderedDict()
        self.ids_to_tokens = []
        with open(vocab_file) as f:
            counter = 0
            for line in f:
                symbol = line[:-1]
                self.vocab[symbol] = counter
                self.ids_to_tokens.append(symbol)
                counter += 1

        self.pad_token_id = self.vocab[pad_token]
        self.eos_token_id = self.vocab[eos_token]
        self.bos_token_id = self.vocab[bos_token]

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.token_is_piece_type_mask = [1 if symbol.isupper() else 0 for symbol in self.ids_to_tokens]

        if notation in NOTATION_TO_REGEX:
            self.move_regex = NOTATION_TO_REGEX[notation]
        else:
            raise ValueError(f"Notation {notation} not supported by tokenizer")
        self.move_pattern = re.compile(self.move_regex)

    def get_vocab(self):
        return self.vocab

    def encode_token(self, token):
        return self.vocab[token]

    def decode_token(self, token_idx):
        return self.ids_to_tokens[token_idx]

    def encode(self, game_str, add_special_tokens=True, get_move_end_positions=True, **kwargs):
        instance = []
        for part in self.move_pattern.split(game_str.strip()):
            if part == '':
                continue
            elif part == ' ':
                instance.append(part)
            else:
                instance.append(part.strip())

        if add_special_tokens:
            instance = [self.bos_token, ' '] + instance + [' ', self.eos_token]   # Empty string denotes end of move
        instance = [None if symbol == ' ' else self.vocab[symbol] for symbol in instance]
        # print(instance)
        if not get_move_end_positions:
            return [symbol for symbol in instance if symbol is not None]
        else:
            end_positions = []
            for idx, symbol in enumerate(instance):
                if symbol is None:
                    end_positions[-1] = 1
                else:
                    end_positions.append(0)

            instance = [symbol for symbol in instance if symbol is not None]  # Remove end of move indicated by None
            assert(len(instance) == len(end_positions))
            return instance, end_positions

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token)

    def save_vocabulary(self, save_directory):
        vocab_file = os.path.join(save_directory, "vocab.txt")
        index = 0
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def __call__(self, lines, max_length=None, **kwargs):
        encoded_inputs = []
        end_positions_list = []
        for line in lines:
            encoded_input, end_positions = self.encode(line, get_move_end_positions=True)
            encoded_inputs.append(encoded_input)
            end_positions_list.append(end_positions)

        max_batch_len = max([len(encoded_input) for encoded_input in encoded_inputs])
        encoded_inputs = [
            encoded_input + [self.pad_token_id] * (max_batch_len - len(encoded_input))
            for encoded_input in encoded_inputs
        ]

        end_positions_list = [
            end_positions + [self.pad_token_id] * (max_batch_len - len(end_positions))
            for end_positions in end_positions_list
        ]

        return {"input_ids": encoded_inputs, "end_positions": end_positions_list}
