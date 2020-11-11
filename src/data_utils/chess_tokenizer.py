import re
from os import path
from constants import NOTATION_TO_REGEX


class ChessTokenizer(object):
    def __init__(self, vocab_dir, pad_symbol='<pad>', start_symbol='<s>', end_symbol='</s>',
                 notation='uci'):
        self.vocab = {}
        self.id2symbol = []
        with open(path.join(vocab_dir, "vocab.txt")) as f:
            counter = 0
            for line in f:
                symbol = line[:-1]
                self.vocab[symbol] = counter
                self.id2symbol.append(symbol)

                counter += 1

        self.pad_token_id = self.vocab[pad_symbol]
        self._pad_token = pad_symbol
        self.bos_token_id = self.vocab[start_symbol]
        self.eos_token_id = self.vocab[end_symbol]
        # Represent end of move symbol by None
        self.vocab[' '] = None

        if notation in NOTATION_TO_REGEX:
            self.move_regex = NOTATION_TO_REGEX[notation]
        else:
            raise ValueError(f"Notation {notation} not supported by tokenizer")
        self.move_pattern = re.compile(self.move_regex)

    def get_vocab(self):
        return self.vocab

    def encode(self, game_str, add_special_tokens=True, get_move_end_positions=True):
        instance = []
        for part in self.move_pattern.split(game_str.strip()):
            if part == '':
                continue
            elif part == ' ':
                instance.append(part)
            else:
                instance.append(part.strip())

        if add_special_tokens:
            instance = ['<s>', ' '] + instance + [' ', '</s>']   # Empty string denotes end of move
        instance = [self.vocab[symbol] for symbol in instance]
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

    def decode(self, id_list, keep_special_tokens=True):
        game = ''.join([self.id2symbol[idx] for idx in id_list])
        if not keep_special_tokens:
            if '<s>' in game:
                game = game[3:]
            if '</s>' in game:
                game = game[:-4]

        return game

    def __call__(self, lines, max_length=None, **kwargs):
        encoded_inputs = []
        end_positions_list = []
        for line in lines:
            encoded_input, end_positions = self.encode(line, get_move_end_positions=True)
            encoded_inputs.append(encoded_input)
            end_positions_list.append(end_positions)

        # if max_length is not None:
        #     encoded_inputs = [encoded_input[:max_length] for encoded_input in encoded_inputs]
        #

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
