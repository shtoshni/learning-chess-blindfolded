import os
import logging
import argparse
import re

from os import path
from collections import Counter

from chess_utils.detection_utils import detect_move_notation
from constants import NOTATION_TO_REGEX

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_dir", type=str, help="Vocab directory.")
    parser.add_argument("--source_file", type=str, help="Source file.")

    args = parser.parse_args()
    assert(path.exists(args.source_file))
    return args


def create_vocab(source_file):
    vocab = Counter(['<pad>', '<s>', '</s>'])
    pred_notation = detect_move_notation(source_file)
    logger.info(f"Predicted notation: {pred_notation}")

    # Get corresponding split pattern
    move_pattern = re.compile(NOTATION_TO_REGEX[pred_notation])
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
    return vocab, pred_notation


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


if __name__ == '__main__':
    parsed_args = parse_args()
    output = create_vocab(parsed_args.source_file)
    save_vocab(output[0], output[1], parsed_args.vocab_dir)