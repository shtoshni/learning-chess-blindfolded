import os
import logging
import argparse
import re

from os import path
from collections import Counter

from constants import NOTATION_TO_REGEX, PIECE_TYPES

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s: %(message)s")


def create_vocab(source_file, add_piece_type, notation="uci"):
    special_symbols = ['<pad>', '<s>', '</s>']
    if add_piece_type:
        special_symbols.extend(PIECE_TYPES)
    vocab = Counter(special_symbols)

    # Get corresponding split pattern
    move_pattern = re.compile(NOTATION_TO_REGEX[notation])
    with open(source_file) as f:
        for line in f:
            game = line.strip()
            move_parts = move_pattern.split(game)
            for part in move_parts:
                part = part.strip()
                if part == '':
                    continue
                else:
                    vocab[part] += 1

    logger.info(f"Size of vocab: {len(vocab)}")
    logger.info(f"{vocab.keys()}")
    return vocab


def save_vocab(vocab, notation, vocab_dir):
    """
    Save vocabulary.
    Args:
        vocab (dict)
        notation (str)
        vocab_dir (path)
    """
    notation_vocab_dir = path.join(vocab_dir, notation)
    if not path.exists(notation_vocab_dir):
        os.makedirs(notation_vocab_dir)

    vocab_file = path.join(notation_vocab_dir, "vocab.txt")
    with open(vocab_file, 'w') as f:
        for token in vocab.keys():
            f.write(token + '\n')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_dir", type=str, help="Vocab directory.")
    parser.add_argument("--source_file", type=str, help="Source file.")
    parser.add_argument("--notation", default="uci", type=str, help="Notation.")
    parser.add_argument("--no_piece_type", default=True, action="store_false", dest="add_piece_type",
                        help="Add piece types to vocab.")

    args = parser.parse_args()
    assert(path.exists(args.source_file))
    return args


if __name__ == '__main__':
    parsed_args = parse_args()
    vocab = create_vocab(parsed_args.source_file, parsed_args.add_piece_type, notation=parsed_args.notation)
    save_vocab(vocab, parsed_args.notation, parsed_args.vocab_dir)
